from __future__ import print_function
from ctypes import CFUNCTYPE, POINTER, c_double, c_float, c_void_p, c_int64
from numba import carray, cfunc, njit
from time import perf_counter as time
from numerous import config
import faulthandler
import numpy as np

import llvmlite.ir as ll
import llvmlite.binding as llvm

faulthandler.enable()
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
llvmmodule = llvm.parse_assembly("")
target_machine = llvm.Target.from_default_triple().create_target_machine()
ee = llvm.create_mcjit_compiler(llvmmodule, target_machine)


class LLVMGenerator:

    def __init__(self, variables, variable_values, n_var):
        self.detailed_print('target data: ', target_machine.target_data)
        self.detailed_print('target triple: ', target_machine.triple)
        self.module = ll.Module()
        self.ext_funcs = {}
        # Define the overall function
        self.fnty = ll.FunctionType(ll.DoubleType().as_pointer(), [
            ll.DoubleType().as_pointer()
        ])
        self.index0 = 0
        self.fnty.args[0].name = "y"

        self.max_var = np.int64(n_var)

        self.variables = variables

        self.var_global = ll.GlobalVariable(self.module, ll.ArrayType(ll.DoubleType(), self.max_var), 'global_var')
        self.var_global.initializer = ll.Constant(ll.ArrayType(ll.DoubleType(), self.max_var),
                                                  [float(v) for v in variable_values])
        ##Fixed structure of the kernel
        func = ll.Function(self.module, self.fnty, name="kernel")
        self.bb_entry = func.append_basic_block(name='entry')
        self.bb_loop = func.append_basic_block(name='main')
        self.bb_store = func.append_basic_block(name='store')
        self.bb_exit = func.append_basic_block(name='exit')
        self.builder = ll.IRBuilder()
        self.builder.position_at_end(self.bb_entry)
        self.index0 = ll.IntType(64)(0)

        self.values = {}

    def add_external_function(self, function):
        """
        Wrap the function and make it available in the LLVM module
        """
        f_c = cfunc(sig=function['signature'])(function['func'])

        if not 'name' in function:
            function['name'] = function['func'].__qualname__

        name = function['name']

        f_c_sym = llvm.add_symbol(name, f_c.address)

        fnty_c_func = ll.FunctionType(ll.VoidType(),
                                      [ll.DoubleType() for _ in function['args']] + [ll.DoubleType().as_pointer() for _
                                                                                     in
                                                                                     function['targets']])
        fnty_c_func.as_pointer(f_c_sym)
        f_llvm = ll.Function(self.module, fnty_c_func, name=name)

        self.ext_funcs[name] = f_llvm

    def generate(self, program, functions, ix_d, n_deriv):
        for function in functions:
            self.add_external_function(function)

        # Define global variable array
        self.builder.branch(self.bb_loop)
        self.builder.position_at_end(self.bb_loop)

        single_arg_sum_counter = 0
        for ix, p in enumerate(program):

            if 'args' in p:
                args = [self.builder.load(self.values[a], 'arg_' + a) for a in p['args']]

            target_pointers = []
            if 'targets' in p:

                for t in p['targets']:
                    index = ll.IntType(64)(self.variables.index(t))

                    ptr = self.var_global
                    indices = [self.index0, index]

                    eptr = self.builder.gep(ptr, indices, name=t)

                    target_pointers.append(eptr)
                    self.values[t] = eptr

            if p['func'] == 'call':

                self.builder.call(self.ext_funcs[p['ext_func']], args + target_pointers)

            elif p['func'] == 'sum':
                la = len(p['args'])
                if la == 0:
                    pass
                elif la == 1:
                    single_arg_sum_counter += 1
                    for t in p['targets']:
                        self.builder.store(args[0], self.values[t])
                elif la == 2:
                    sum_ =  self.builder.fadd(args[0], args[1])
                    for t in p['targets']:
                        self.builder.store(sum_, self.values[t])

                else:
                    accum = self.builder.fadd(args[0], args[1])
                    for i, a in enumerate(args[2:]):
                        accum = self.builder.fadd(accum, a)

                    for t in p['targets']:
                        self.builder.store(accum, self.values[t])

        self.detailed_print('single assignments: ', single_arg_sum_counter)

        self.builder.branch(self.bb_store)
        self.builder.position_at_end(self.bb_store)

        if len(program) > 0:
            raise ValueError(f'Still program lines left: {str(program)}')

        self.builder.branch(self.bb_exit)

        self.builder.position_at_end(self.bb_exit)

        indexd = ll.IntType(64)(ix_d)

        dg_ptr = self.builder.gep(self.var_global, [self.index0, indexd])

        self.builder.ret(dg_ptr)
        # Vars function

        self.save_module('llvm_IR_code.txt')

        llmod = llvm.parse_assembly(str(self.module))

        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 1
        pm = llvm.create_module_pass_manager()
        pmb.populate(pm)

        pm.run(llmod)

        # self.detailed_print("-- optimize:", t6 - t5)

        ee.add_module(llmod)

        ee.finalize_object()
        cfptr = ee.get_function_address("kernel")
        cfptr_var = ee.get_function_address("vars")


        c_float_type = type(np.ctypeslib.as_ctypes(np.float64()))
        self.detailed_print(c_float_type)

        diff_ = CFUNCTYPE(POINTER(c_float_type), POINTER(c_float_type))(cfptr)

        vars_ = CFUNCTYPE(POINTER(c_float_type), c_int64)(cfptr_var)

        @njit('float64[:](float64[:])')
        def diff(y):

            deriv_pointer = diff_(y.ctypes)
            return carray(deriv_pointer, (n_deriv,)).copy()

        @njit('float64[:](float64)')
        def variables__(f):
            variables_pointer = vars_(0)
            variables_array = carray(variables_pointer, (self.max_var,))

            return variables_array.copy()

        return diff, variables__

    def detailed_print(self, *args, sep=' ', end='\n', file=None):
        if config.PRINT_LLVM:
            print(*args, sep, end, file)

    def save_module(self, filename):
        with open(filename, 'w') as f:
            f.write(str(self.module))

    def add_load(self, sequence, name):
        for ix, v in enumerate(sequence):
            index = ll.IntType(64)(ix)

            if name == 'variables':
                ptr = self.var_global
                indices = [self.index0, index]

            elif name == 'y':
                ptr = self.func.args[0]
                indices = [index]

            eptr = self.builder.gep(ptr, indices, name + "_" + v)
            self.values[v] = eptr

    def add_store(self, sequence, name):
        for ix, v in enumerate(sequence):
            index = ll.IntType(64)(ix)

            if name == 'variables':
                ptr = self.var_global
                indices = [self.index0, index]

            eptr = self.builder.gep(ptr, indices)
            self.builder.store(self.builder.load(self.values[v]), eptr)

    def add_call(self):
        pass


    def build_var(self):
        fnty_vars = ll.FunctionType(ll.DoubleType().as_pointer(), [ll.IntType(64)])

        vars_func = ll.Function(self.module, fnty_vars, name="vars")

        bb_entry_var = vars_func.append_basic_block()

        bb_exit_var = vars_func.append_basic_block()

        builder_var = ll.IRBuilder()
        builder_var.position_at_end(bb_entry_var)
        builder_var.branch(bb_exit_var)

        builder_var.position_at_end(bb_exit_var)

        index0_var = ll.IntType(64)(0)
        vg_ptr = builder_var.gep(self.var_global, [index0_var, index0_var])
        builder_var.ret(vg_ptr)


    def add_mapping(self):
        pass

    def add_set_call(self):
        pass

    def add_set_mapping(self):
        pass
