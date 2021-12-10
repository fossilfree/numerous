from __future__ import print_function

import os
from ctypes import CFUNCTYPE, POINTER, c_double, c_void_p, c_int64
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

LISTING_FILEPATH = "tmp/listings/"
LISTINGFILENAME = "_llvm_listing.ll"
LLVMOPTLISTING_FILENAME = "tmp/listings/llvm_listing_opt.ll"

class LLVMBuilder:
    """
    Building an LLVM module.
    """

    def __init__(self, initial_values, variable_names, states, derivatives, system_tag=""):
        """
        initial_values - initial values of global variables array. Array should be ordered in  such way
         that all the derivatives are located in the tail.
        variable_names - dictionary
        states - states
        derivatives -  derivatives
        """

        self.loopcount = 0
        self.detailed_print('target data: ', target_machine.target_data)
        self.detailed_print('target triple: ', target_machine.triple)
        self.module = ll.Module()
        self.ext_funcs = {}
        self.states = states
        self.derivatives = derivatives
        self.n_deriv = len(derivatives)
        # Define the overall function
        self.fnty = ll.FunctionType(ll.DoubleType().as_pointer(), [
            ll.DoubleType().as_pointer()
        ])
        self.index0 = 0
        self.fnty.args[0].name = "y"
        self.system_tag = system_tag
        self.variable_names = {}

        for k, v in variable_names.items():
            self.variable_names[k] = v
            self.variable_names[v] = k


        self.max_var = initial_values.shape[0]

        self.var_global = ll.GlobalVariable(self.module, ll.ArrayType(ll.DoubleType(), self.max_var), 'global_var')
        self.var_global.initializer = ll.Constant(ll.ArrayType(ll.DoubleType(), self.max_var),
                                                  [float(v) for v in initial_values])

        self.deriv_global = ll.GlobalVariable(self.module, ll.ArrayType(ll.DoubleType(), self.n_deriv), 'derivatives')
        self.deriv_global.initializer = ll.Constant(ll.ArrayType(ll.DoubleType(), self.n_deriv),
                                                  [float(0) for _ in range(self.n_deriv)])

        # Creatinf the fixed structure of the kernel
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
        for idx,state in enumerate(self.states):
            self.load_state_variable(idx, state)

        # go through all states to put them in store block
        for state in self.states:
            self.store_variable(state)

        self.builder.position_at_end(self.bb_entry)
        self.builder.branch(self.bb_loop)
        self.builder.position_at_end(self.bb_loop)

    def add_external_function(self, function, signature, number_of_args, target_ids,replacements=None,replace_name=None):
        """
        Wrap the function and make it available in the LLVM module
        """

        f_c = cfunc(sig=signature)(function)
        name = function.__qualname__
        f_c_sym = llvm.add_symbol(name, f_c.address)
        llvm_signature = np.tile(ll.DoubleType(), number_of_args).tolist()
        for i in target_ids:
            llvm_signature[i] = ll.DoubleType().as_pointer()

        fnty_c_func = ll.FunctionType(ll.VoidType(),
                                      llvm_signature)
        fnty_c_func.as_pointer(f_c_sym)
        f_llvm = ll.Function(self.module, fnty_c_func, name=name)

        self.ext_funcs[name] = f_llvm

    def generate(self, imports=None, system_tag=None, save_to_file=False):

        self.builder.position_at_end(self.bb_loop)

        self.builder.branch(self.bb_store)
        self.builder.position_at_end(self.bb_store)

        self.builder.branch(self.bb_exit)

        self.builder.position_at_end(self.bb_exit)

        for i, deriv in enumerate(self.derivatives):
            if deriv not in self.values:
                self._create_target_pointer(deriv)
            pointer = self._get_target_pointers([deriv])
            dg_ptr = self.builder.gep(self.deriv_global, [ll.IntType(64)(0), ll.IntType(64)(i)])
            self.builder.store(self.builder.load(*pointer), dg_ptr)

        self.builder.ret(self.builder.gep(self.deriv_global, [self.index0, self.index0]))

        # build vars read function
        self._build_var_r()

        # build vars write function
        self._build_var_w()

        if save_to_file:
            self.save_module(LISTING_FILEPATH+self.system_tag+LISTINGFILENAME)

        llmod = llvm.parse_assembly(str(self.module))

        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 1
        pm = llvm.create_module_pass_manager()
        pmb.populate(pm)

        pm.run(llmod)
        os.makedirs(os.path.dirname(LLVMOPTLISTING_FILENAME), exist_ok=True)
        if save_to_file:
            with open(LLVMOPTLISTING_FILENAME, 'w') as f:
                f.write(str(llmod))

        ee.add_module(llmod)

        ee.finalize_object()
        cfptr = ee.get_function_address("kernel")

        cfptr_var_r = ee.get_function_address("vars_r")
        cfptr_var_w = ee.get_function_address("vars_w")

        c_float_type = type(np.ctypeslib.as_ctypes(np.float64()))
        self.detailed_print(c_float_type)

        diff_ = CFUNCTYPE(POINTER(c_float_type), POINTER(c_float_type))(cfptr)

        vars_r = CFUNCTYPE(POINTER(c_float_type), c_int64)(cfptr_var_r)

        vars_w = CFUNCTYPE(c_void_p, c_double, c_int64)(cfptr_var_w)

        n_deriv = self.n_deriv

        @njit('float64[:](float64[:])')
        def diff(y):
            deriv_pointer = diff_(y.ctypes)
            return carray(deriv_pointer, (n_deriv,)).copy()

        max_var = self.max_var

        @njit('float64[:]()')
        def read_variables():
            variables_pointer = vars_r(0)
            variables_array = carray(variables_pointer, (max_var,))

            return variables_array.copy()

        @njit('void(float64,int64)')
        def write_variables(var, idx):
            vars_w(var, idx)

        return diff, read_variables, write_variables
        # write_variables

    def detailed_print(self, *args, sep=' ', end='\n', file=None):
        if config.PRINT_LLVM:
            print(*args, sep, end, file)

    def save_module(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(str(self.module))

    def load_global_variable(self, variable_name):
        index = ll.IntType(64)(self.variable_names[variable_name])
        ptr = self.var_global
        indices = [self.index0, index]
        eptr = self.builder.gep(ptr, indices, name="variable_" + variable_name)
        self.values[variable_name] = eptr

    def load_state_variable(self,idx, state_name):
        index_args = ll.IntType(64)(idx)
        ptr = self.func.args[0]
        indices = [index_args]
        eptr = self.builder.gep(ptr, indices, name="state_" + state_name)
        self.values[state_name] = eptr
        ptr = self.var_global
        index_global = ll.IntType(64)(self.variable_names[state_name])
        indices = [self.index0, index_global]
        eptr = self.builder.gep(ptr, indices)
        self.builder.store(self.builder.load(self.values[state_name]), eptr)

    def store_variable(self, variable_name):
        _block = self.builder.block
        self.builder.position_at_end(self.bb_store)
        index = ll.IntType(64)(self.variable_names[variable_name])
        ptr = self.var_global
        indices = [self.index0, index]
        eptr = self.builder.gep(ptr, indices)
        self.builder.store(self.builder.load(self.values[variable_name]), eptr)
        self.builder.position_at_end(_block)

    def add_call(self, external_function_name, args, target_ids):
        arg_pointers = []
        for i1, arg in enumerate(args):
            if i1 in target_ids:
                if arg not in self.values:
                    self._create_target_pointer(arg)
                arg_pointers.append(*self._get_target_pointers([arg]))
            else:
                arg_pointers.append(*self._load_args([arg]))
        self.builder.position_at_end(self.bb_loop)
        self.builder.call(self.ext_funcs[external_function_name], arg_pointers)

    def _build_var_r(self):
        fnty_vars = ll.FunctionType(ll.DoubleType().as_pointer(), [ll.IntType(64)])

        vars_func = ll.Function(self.module, fnty_vars, name="vars_r")

        bb_entry_var = vars_func.append_basic_block()

        bb_exit_var = vars_func.append_basic_block()

        builder_var = ll.IRBuilder()
        builder_var.position_at_end(bb_entry_var)
        builder_var.branch(bb_exit_var)

        builder_var.position_at_end(bb_exit_var)

        index0_var = ll.IntType(64)(0)
        vg_ptr = builder_var.gep(self.var_global, [index0_var, index0_var])
        builder_var.ret(vg_ptr)

    def _build_var_w(self):
        fnty_vars = ll.FunctionType(ll.VoidType(), [ll.DoubleType(), ll.IntType(64)])
        fnty_vars.args[0].name = "vars"
        fnty_vars.args[1].name = "idx"
        vars_func = ll.Function(self.module, fnty_vars, name="vars_w")
        bb_entry_var = vars_func.append_basic_block()
        bb_exit_var = vars_func.append_basic_block()

        builder_var = ll.IRBuilder()
        builder_var.position_at_end(bb_entry_var)
        builder_var.branch(bb_exit_var)
        builder_var.position_at_end(bb_exit_var)

        index = vars_func.args[1]
        ptr = self.var_global
        indices = [self.index0, index]
        eptr = builder_var.gep(ptr, indices)
        builder_var.store(vars_func.args[0], eptr)

        builder_var.ret_void()

    def _store_to_global_target(self, target, value):
        if target not in self.values:
            self._create_target_pointer(target)
        self.builder.store(value, self.values[target])

    def add_mapping(self, args, targets):
        self.builder.position_at_end(self.bb_loop)
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

    def _load_args(self, args):
        loaded_args = []
        self.builder.position_at_end(self.bb_loop)
        for a in args:
            if a not in self.values:
                self.load_global_variable(a)
            loaded_args.append(self.builder.load(self.values[a], 'arg_' + a))
        return loaded_args

    def _get_target_pointers(self, targets):
        return [self.values[target] for target in targets]

    def _create_target_pointer(self, t):
        index = ll.IntType(64)(self.variable_names[t])
        ptr = self.var_global
        indices = [self.index0, index]
        eptr = self.builder.gep(ptr, indices, name=t)
        self.values[t] = eptr

    ##Should be moved to utils
    def _is_range_and_ordered(self, a_list):
        e = len(a_list)
        i = a_list[0]
        return all(ele >= i and ele < i + e for ele in a_list) and all(a_list[i] <= a_list[i + 1] for i in range(e - 1))

    def add_set_call(self, external_function_name, variable_name_arg_and_trg, targets_ids):
        ## TODO check allign
        self.builder.position_at_end(self.bb_loop)
        number_of_ix_arg = len(variable_name_arg_and_trg[0])
        number_of_ix_trg = len(targets_ids)
        size = len(variable_name_arg_and_trg)
        # order_arg = [self.variable_names[x] for x in list(itertools.chain(*variable_name_arg))]
        # order_trg = [self.variable_names[x] for x in list(itertools.chain(*variable_name_trg))]
        # if not ((self._is_range_and_ordered(order_arg)) and (self._is_range_and_ordered(order_trg))):
        #     raise ValueError("llvm set call arguments are not align.")

        index_arg = (ll.IntType(64)(self.variable_names[variable_name_arg_and_trg[0][0]] - 1))
        index_arg_ptr = self.builder.alloca(ll.IntType(64))
        self.builder.store(index_arg, index_arg_ptr)

        ##crete new block

        size_ptr = ll.IntType(64)(size)

        loop_cond = self.func.append_basic_block(name='for' + str(self.loopcount) + '.cond')
        self.builder.position_at_end(loop_cond)
        i0 = self.builder.phi(ll.IntType(64), name='loop' + str(self.loopcount) + '.ix')

        self.builder.position_at_end(self.bb_loop)
        self.builder.branch(loop_cond)

        loop_body = self.func.append_basic_block(name='for' + str(self.loopcount) + '.body')
        self.builder.position_at_end(loop_body)

        loaded_args = []
        loaded_trgs = []
        for i1 in range(number_of_ix_arg):
            index_arg = self.builder.load(index_arg_ptr)
            index_arg = self.builder.add(index_arg, ll.IntType(64)(1), name="index_arg")
            self.builder.store(index_arg, index_arg_ptr)

            indices_arg = [self.index0, index_arg]
            eptr_arg = self.builder.gep(self.var_global, indices_arg, name="loop_" + str(self.loopcount) + " _arg_")
            if i1 in targets_ids:
                loaded_args.append(eptr_arg)
            else:
                loaded_args.append(self.builder.load(eptr_arg, 'arg_' + str(self.loopcount) + "_" + str(i1)))

        # EQ_system_SET_oscillators_mechanics_llvm
        self.builder.call(self.ext_funcs[external_function_name], loaded_args)

        loop_inc = self.func.append_basic_block(name='for' + str(self.loopcount) + '.inc')
        self.builder.position_at_end(loop_inc)
        result = self.builder.add(i0, ll.IntType(64)(1), name="res")
        self.builder.branch(loop_cond)

        loop_end = self.func.append_basic_block(name='main' + str(self.loopcount))

        self.builder.position_at_end(loop_cond)
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
