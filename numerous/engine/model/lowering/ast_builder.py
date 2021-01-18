from __future__ import print_function

import itertools
from ctypes import CFUNCTYPE, POINTER, c_double, c_void_p, c_int64
from numba import carray, cfunc, njit

from numerous.engine.model.utils import wrap_function
from numerous import config
import faulthandler
import numpy as np
import logging
import ast

from numerous.engine.model.lowering.utils import generate_code_file

GLOBAL_ARRAY = ast.Name(id="kernel_variables")


class ASTBuilder:
    """
    Building an AST module.
    """

    def __init__(self, initial_values, variable_names, states, derivatives):
        """
        initial_values - initial values of global variables array. Array should be ordered in  such way
         that all the derivatives are located in the tail.
        variable_names - dictionary
        states - states
        derivatives -  derivatives
        """
        self.body_init_set_var = [ast.Assign(targets=[GLOBAL_ARRAY],
                                             value=ast.Call(func=ast.Name(id='np.array',
                                                                          ctx=ast.Load()),
                                                            args=[ast.List(
                                                                elts=[ast.Constant(value=v) for v in initial_values],
                                                                ctx=ast.Load())],
                                                            keywords=[]))]

        self.kernel_function = []
        self.variable_names = variable_names
        self.states = states
        self.derivatives = derivatives
        self.functions = []
        self.defined_functions = []
        self.body = []

        self.read_args_section = [ast.Expr(value=ast.Call(func=ast.Name(id='np.put'),
                                                          args=[GLOBAL_ARRAY,
                                                                ast.List(
                                                                    elts=[ast.Constant(value=v) for v in
                                                                          [self.variable_names[x] for x in
                                                                           self.states]],
                                                                    ctx=ast.Load()),
                                                                ast.Name(id='states', ctx=ast.Store())
                                                                ],
                                                          keywords=[]))]

        self.return_section = [ast.Return(
            value=ast.Call(func=ast.Name(id='np.take'), args=[GLOBAL_ARRAY,
                                                              ast.List(
                                                                  elts=[ast.Constant(value=v) for v in
                                                                        [self.variable_names[x] for x in
                                                                         self.derivatives]],
                                                                  ctx=ast.Load())
                                                              ],
                           keywords=[]))]

    def add_external_function(self, function, signature, number_of_args, target_ids):
        self.functions.append(function)
        self.defined_functions.append(function.name)

    def generate(self, filename=None, save_opt=False):
        kernel = wrap_function('global_kernel', self.read_args_section + self.body + self.return_section, decorators=[],
                               args=ast.arguments(args=[ast.arg(arg="states",
                                                                annotation=None)], vararg=None, defaults=[],
                                                  kwarg=None))

        generate_code_file([x for x in self.functions] + self.body_init_set_var + [kernel], "kernel.py")

    def detailed_print(self, *args, sep=' ', end='\n', file=None):
        if config.PRINT_LLVM:
            print(*args, sep, end, file)

    def store_variable(self, variable_name):
        pass

    def add_call(self, external_function_name, args, target_ids):

        arg_ids = np.arange(len(args))
        start_idx = self.variable_names[args[0]]
        targets = []
        args = []
        for target_id in target_ids:
            targets.append(
                ast.Subscript(value=GLOBAL_ARRAY, slice=ast.Index(value=ast.Constant(value=target_id + start_idx)),
                              ctx=ast.Store()))
        for arg_id in arg_ids:
            args.append(
                ast.Subscript(value=GLOBAL_ARRAY, slice=ast.Index(value=ast.Constant(value=arg_id + start_idx)),
                              ctx=ast.Load))
        self.body.append(ast.Assign(targets=[ast.Tuple(elts=targets)],
                                    value=ast.Call(func=ast.Name(id=external_function_name, ctx=ast.Load()),
                                                   args=args, keywords=[])))


    def add_mapping(self, args, target):
        if len(target) > 1:
            raise ValueError("Only mapping to single target is supported")
        arg_idxs = []
        for arg in args:
            arg_idxs.append(self.variable_names[arg])
        target_idx = self.variable_names[target[0]]
        print(target_idx)
        if len(args) == 1:
            self.body.append(ast.Assign(targets=[ast.Subscript(value=GLOBAL_ARRAY,
                                                               slice=ast.Index(
                                                                   value=ast.Constant(value=target_idx, kind=None)))],
                                        value=ast.Subscript(value=GLOBAL_ARRAY,
                                                            slice=ast.Index(
                                                                value=ast.Constant(value=arg_idxs[0], kind=None)))))
        else:
            self.body.append(ast.Assign(targets=[ast.Subscript(value=GLOBAL_ARRAY,
                                                               slice=ast.Index(
                                                                   value=ast.Constant(value=target_idx, kind=None)))],
                                        value=ast.BinOp(left=self._generate_sum_left(arg_idxs[1:]), op=ast.Add(),
                                                        right=ast.Subscript(value=GLOBAL_ARRAY,
                                                                            slice=ast.Index(
                                                                                value=ast.Constant(value=arg_idxs[0],
                                                                                                   kind=None))))
                                        ))

    def _generate_sum_left(self, arg_idxs):
        if len(arg_idxs) > 0:
            return ast.BinOp(left=self._generate_sum_left(arg_idxs[1:]), op=ast.Add(),
                             right=ast.Subscript(value=GLOBAL_ARRAY,
                                                 slice=ast.Index(
                                                     value=ast.Constant(value=arg_idxs[0], kind=None))))
        else:
            ## adding 0 if no summation arguments left
            return ast.Constant(value=0, kind=None)

    def add_set_call(self, external_function_name, variable_name_arg_and_trg, targets_ids):
        # Module(body=[For(target=Name(id='x', ctx=Store()),
        #                  iter=Call(func=Name(id='range', ctx=Load()), args=[Constant(value=10, kind=None)],
        #                            keywords=[]), body=[Expr(value=Constant(value=Ellipsis, kind=None))],
        #                  orelse=[Expr(value=Constant(value=Ellipsis, kind=None))], type_comment=None)], type_ignores=[])
        #
        # pass
        self.body.append(ast.For(target=ast.Name(id='i', ctx=ast.Store()),
                                 iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=
                                 [ast.Constant(value=len(variable_name_arg_and_trg))], keywords=[]),
                                 body=[self._generate_set_call_body(external_function_name,
                                                                    len(variable_name_arg_and_trg[0]),
                                                                    self.variable_names[
                                                                        variable_name_arg_and_trg[0][0]],
                                                                    targets_ids)],
                                 orelse=[]))

    def _generate_set_call_body(self, external_function_name, arg_length, strart_idx, target_ids):
        arg_ids = np.arange(arg_length)
        targets = []
        args = []

        for target_id in target_ids:
            targets.append(ast.Subscript(value=GLOBAL_ARRAY,
                                         slice=ast.Index(
                                             value=ast.BinOp(
                                                 left=ast.Constant(value=target_id + strart_idx),
                                                 op=ast.Add(),
                                                 right=ast.BinOp(left=ast.Constant(value=arg_length, kind=None),
                                                                 op=ast.Mult(),
                                                                 right=ast.Name(id='i', ctx=ast.Load())))),
                                         ctx=ast.Store()))

        for arg_id in arg_ids:
            args.append(
                ast.Subscript(value=GLOBAL_ARRAY, slice=ast.Index(value=ast.BinOp(
                                                 left=ast.Constant(value=arg_id + strart_idx),
                                                 op=ast.Add(),
                                                 right=ast.BinOp(left=ast.Constant(value=arg_length, kind=None),
                                                                 op=ast.Mult(),
                                                                 right=ast.Name(id='i', ctx=ast.Load())))),
                              ctx=ast.Load))
        if len(targets) > 1:
            return ast.Assign(targets=[ast.Tuple(elts=targets)],
                              value=ast.Call(func=ast.Name(id=external_function_name, ctx=ast.Load()),
                                             args=args, keywords=[]))
        else:
            return ast.Assign(targets=[targets[0]],
                              value=ast.Call(func=ast.Name(id=external_function_name, ctx=ast.Load()),
                                             args=args, keywords=[]))

    def add_set_mapping(self, args2d, targets2d):
        pass
