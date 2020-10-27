from __future__ import print_function
from ctypes import CFUNCTYPE, POINTER, c_double, c_float, c_void_p, c_int64
from numba import carray, cfunc, njit
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


class LLVMBuilder:
    """
    Building an LLVM module.
    """

    def __init__(self, initial_values, variable_names, number_of_states, number_of_derivatives):
        """
        initial_values - initial values of global variables array. Array should be ordered in  such way
         that all the derivatives are located in the tail.
        variable_names - dictionary
        number_of_states - number of states
        number_of_derivatives -  number of derivatives
        """

        self.loopcount = 0
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

        self.variable_names = {}

        for k, v in variable_names.items():
            self.variable_names[k] = v
            self.variable_names[v] = k
        self.number_of_states = number_of_states

        self.n_deriv = number_of_derivatives

        self.max_var = initial_values.shape[0]
        self.ix_d = self.max_var - number_of_derivatives

        self.var_global = ll.GlobalVariable(self.module, ll.ArrayType(ll.DoubleType(), self.max_var), 'global_var')
        self.var_global.initializer = ll.Constant(ll.ArrayType(ll.DoubleType(), self.max_var),
                                                  [float(v) for v in initial_values])
        ##Creatinf the fixed structure of the kernel
        self.func = ll.Function(self.module, self.fnty, name="kernel")
        self.bb_entry = self.func.append_basic_block(name='entry')
        self.bb_loop = self.func.append_basic_block(name='main')
        self.bb_store = self.func.append_basic_block(name='store')
        self.bb_exit = self.func.append_basic_block(name='exit')
        self.builder = ll.IRBuilder()
        self.builder.position_at_end(self.bb_entry)
        self.index0 = ll.IntType(64)(0)

        self.values = {}

        # go through  all states to put them in load block
        for i in range(self.number_of_states):
            self.load_state_variable(self.variable_names[i])

        # go through all states to put them in store block
        for i in range(self.number_of_states):
            self.store_variable(self.variable_names[i])

    def add_external_function(self, function, signature, args_count, targets_count):
        """
        Wrap the function and make it available in the LLVM module
        """
        f_c = cfunc(sig=signature)(function)

        name = function.__qualname__

        f_c_sym = llvm.add_symbol(name, f_c.address)
        llvm_signature = np.tile(ll.DoubleType(), args_count).tolist() + np.tile(ll.DoubleType().as_pointer(),
                                                                                 targets_count).tolist()
        fnty_c_func = ll.FunctionType(ll.VoidType(),
                                      llvm_signature)
        fnty_c_func.as_pointer(f_c_sym)
        f_llvm = ll.Function(self.module, fnty_c_func, name=name)

        self.ext_funcs[name] = f_llvm

    def generate(self, filename):

        self.builder.position_at_end(self.bb_entry)
        self.builder.branch(self.bb_loop)
        self.builder.position_at_end(self.bb_loop)

        # ----
        # self.detailed_print('single assignments: ', single_arg_sum_counter)

        self.builder.branch(self.bb_store)
        self.builder.position_at_end(self.bb_store)

        self.builder.branch(self.bb_exit)

        self.builder.position_at_end(self.bb_exit)

        indexd = ll.IntType(64)(self.ix_d)

        dg_ptr = self.builder.gep(self.var_global, [self.index0, indexd])

        self.builder.ret(dg_ptr)

        # build vars function
        self._build_var()

        self.save_module(filename)

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

        n_deriv = self.n_deriv

        @njit('float64[:](float64[:])')
        def diff(y):
            deriv_pointer = diff_(y.ctypes)
            return carray(deriv_pointer, (n_deriv,)).copy()

        max_var = self.max_var

        @njit('float64[:]()')
        def variables__():
            variables_pointer = vars_(0)
            variables_array = carray(variables_pointer, (max_var,))

            return variables_array.copy()

        return diff, variables__

    def detailed_print(self, *args, sep=' ', end='\n', file=None):
        if config.PRINT_LLVM:
            print(*args, sep, end, file)

    def save_module(self, filename):
        with open(filename, 'w') as f:
            f.write(str(self.module))

    # llvm_sequence += [{'func': 'load', 'ix': ix + lenstates, 'var': v, 'arg': 'variables'} for ix, v in
    #                   enumerate(vars_init[lenstates:])]
    # llvm_sequence += [{'func': 'load', 'ix': ix, 'var': s, 'arg': 'y'} for ix, s in enumerate(states)]

    def load_global_variable(self, variable_name):
        index = ll.IntType(64)(self.variable_names[variable_name])
        ptr = self.var_global
        indices = [self.index0, index]
        eptr = self.builder.gep(ptr, indices, name="variable_" + variable_name)
        self.values[variable_name] = eptr

    def load_state_variable(self, state_name):
        index = ll.IntType(64)(self.variable_names[state_name])
        ptr = self.func.args[0]
        indices = [index]
        eptr = self.builder.gep(ptr, indices, name="state_" + state_name)
        self.values[state_name] = eptr

    def store_variable(self, variable_name):
        _block = self.builder.block
        self.builder.position_at_end(self.bb_store)
        index = ll.IntType(64)(self.variable_names[variable_name])
        ptr = self.var_global
        indices = [self.index0, index]
        eptr = self.builder.gep(ptr, indices)
        self.builder.store(self.builder.load(self.values[variable_name]), eptr)
        self.builder.position_at_end(_block)

    def add_call(self, external_function, args, targets):
        arg_pointers = self._get_arg_pointers(args)
        for target in targets:
            if target not in self.values:
                self._create_target_pointer(target)
        target_pointers = self._get_target_pointers(targets)
        self.builder.position_at_end(self.bb_loop)
        self.builder.call(self.ext_funcs[external_function.__qualname__], arg_pointers + target_pointers)

    def _build_var(self):
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

    def _store_to_global_target(self, target, value):
        if target not in self.values:
            self._create_target_pointer(target)
        self.builder.store(value, self.values[target])

    def _load_args(self, args):
        loaded_args = []
        for a in args:
            if a not in self.values:
                self.load_global_variable(a)
            loaded_args.append(self.builder.load(self.values[a], 'arg_' + a))
        return loaded_args

    def add_mapping(self, args, targets):
        la = len(args)

        loaded_args_ptr = self._load_args(args)

        if la == 1:
            for t in targets:
                self._store_to_global_target(t, loaded_args_ptr[0])
        else:
            accum = self.builder.fadd(loaded_args_ptr[0], loaded_args_ptr[1])
            for i, a in enumerate(loaded_args_ptr[2:]):
                accum = self.builder.fadd(accum, a)

            for t in targets:
                self._store_to_global_target(t, accum)

    def _get_arg_pointers(self, args):
        return [self.builder.load(self.values[a], 'arg_' + a) for a in args]

    def _get_target_pointers(self, targets):
        return [self.values[target] for target in targets]

    def _create_target_pointer(self, t):
        index = ll.IntType(64)(self.variable_names[t])
        ptr = self.var_global
        indices = [self.index0, index]
        eptr = self.builder.gep(ptr, indices, name=t)
        self.values[t] = eptr

    def add_set_call(self, size):
        # function,args2d,targets2d):
        ## TODO check allign
        ##crete new block

        size_ptr = ll.IntType(64)(size)
        inc = self.builder.alloca(ll.IntType(64), name='inc')
        self.builder.store(ll.IntType(64)(0), inc)

        loop_cond = self.func.append_basic_block(name='for' + str(self.loopcount) + '.cond')
        self.builder.position_at_end(loop_cond)

        self.builder.position_at_end(self.bb_loop)
        self.builder.branch(loop_cond)

        loop_body = self.func.append_basic_block(name='for' + str(self.loopcount) + '.body')
        self.builder.position_at_end(loop_body)

        loop_inc = self.func.append_basic_block(name='for' + str(self.loopcount) + '.inc')
        self.builder.position_at_end(loop_inc)
        inc_value = self.builder.load(inc)
        result = self.builder.add(inc_value, ll.IntType(64)(1), name="res")
        self.builder.branch(loop_cond)

        loop_end = self.func.append_basic_block(name='main' + str(self.loopcount))
        self.builder.position_at_end(loop_end)

        self.builder.position_at_end(loop_cond)
        i0 = self.builder.phi(ll.IntType(64), name='loop' + str(self.loopcount) + '.ix')
        i0.add_incoming(ll.IntType(64)(0), self.bb_loop)
        i0.add_incoming(result, loop_inc)
        cmp = self.builder.icmp_unsigned("<", i0, size_ptr, name="cmp")
        self.builder.cbranch(cmp, loop_body, loop_end)

        self.builder.position_at_end(loop_body)
        self.builder.branch(loop_inc)
        self.bb_loop = loop_end

        self.loopcount += 1

    def add_set_mapping(self, args2d, targets2d):
        pass
