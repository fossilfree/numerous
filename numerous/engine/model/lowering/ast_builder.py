from __future__ import print_function

import os
import types

from numerous.engine.model.utils import wrap_function
from numerous import config
import numpy as np
import ast
from copy import deepcopy as copy

from numerous.engine.model.lowering.utils import generate_code_file

GLOBAL_ARRAY = ast.Name(id="kernel_variables")
LISTING_FILEPATH = "tmp/listings/"
LISTINGFILENAME = "_kernel.py"


class ASTBuilder:
    """
    Building an AST module.
    """

    def __init__(self, initial_values, variable_names, states, derivatives, system_tag=""):
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
                                                            keywords=[]
                                                            ), lineno=0)]

        self.kernel_function = []
        self.variable_names = variable_names
        self.states = states
        self.derivatives = derivatives
        self.functions = []
        self.defined_functions = []
        self.body = []
        self.kernel_filename = LISTING_FILEPATH + system_tag + LISTINGFILENAME
        self.cnd_calls={}

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

    def add_external_function(self, function: ast.FunctionDef, signature: str, number_of_args: int,
                              target_ids: list[int]):
        self.functions.append(function)
        self.defined_functions.append(function.name)

    def generate(self, imports, system_tag="", external_functions_source=False, save_to_file=False):
        arguments = [ast.arg(arg="states", annotation=None)]
        defaults=[]

        for i in self.body:
            if hasattr(i, 'cnd') and i.cnd:
                arguments.append(ast.arg(arg=i.body[0].value.func.id+'_flag', annotation=None))
                defaults.append(ast.Constant(i.active))

        kernel = wrap_function('global_kernel', self.read_args_section + self.body + self.return_section,
                               decorators=[],
                               args=ast.arguments(posonlyargs=[], args=arguments, vararg=None,
                                                  defaults=defaults,
                                                  kwarg=None, kwonlyargs=[]))
        variable_names_print = []
        for key, value in self.variable_names.items():
            variable_names_print.append('#' + str(key) + ' : ' + str(value))
        code = generate_code_file([x for x in self.functions] + self.body_init_set_var + [kernel], self.kernel_filename,
                                  imports,
                                  external_functions_source=external_functions_source,
                                  names='\n'.join(variable_names_print) + '\n')

        kernel_module = types.ModuleType('python_kernel')
        if save_to_file:
            os.makedirs(os.path.dirname(self.kernel_filename), exist_ok=True)
            with open(self.kernel_filename, 'w') as f:
                f.write(code)
            exec('from tmp.listings.' + system_tag + '_kernel import *', globals())

            def var_func():
                return kernel_variables

            def var_write(value, idx):
                np.put(kernel_variables, [idx], value)

            return global_kernel, var_func, var_write
        else:
            exec(code, kernel_module.__dict__)

            def var_func():
                return kernel_module.kernel_variables

            def var_write(value, idx):
                np.put(kernel_module.kernel_variables, [idx], value)

            return kernel_module.global_kernel, var_func, var_write,

        return kernel_module.global_kernel, var_func, var_write

    def unparse(self, imports, system_tag="", external_functions_source=False):
        arguments = [ast.arg(arg="states", annotation=None)]
        defaults=[]

        for i in self.body:
            if hasattr(i, 'cnd') and i.cnd:
                arguments.append(ast.arg(arg=i.body[0].value.func.id+'_flag', annotation=None))
                defaults.append(ast.Constant(i.active))

        kernel = wrap_function('global_kernel', self.read_args_section + self.body + self.return_section,
                               decorators=[],
                               args=ast.arguments(posonlyargs=[], args=arguments, vararg=None,
                                                  defaults=defaults,
                                                  kwarg=None, kwonlyargs=[]))
        variable_names_print = []
        for key, value in self.variable_names.items():
            variable_names_print.append('#' + str(key) + ' : ' + str(value))
        code = generate_code_file([x for x in self.functions] + self.body_init_set_var + [kernel], self.kernel_filename,
                                  imports,
                                  external_functions_source=external_functions_source,
                                  names='\n'.join(variable_names_print) + '\n')

        return code

    def detailed_print(self, *args, sep=' ', end='\n', file=None):
        if config.PRINT_LLVM:
            print(*args, sep, end, file)
    
    def store_variable(self, variable_name):
        pass

    def _create_assignments(self, external_function_name, input_args, target_ids):
        arg_ids = map(lambda arg: self.variable_names[arg], input_args)
        targets = []
        args = []
        for target_id in target_ids:
            targets.append(
                ast.Subscript(value=GLOBAL_ARRAY,
                              slice=ast.Index(value=ast.Constant(value=self.variable_names[input_args[target_id]])),
                              ctx=ast.Store()))
        for arg_id in arg_ids:
            args.append(
                ast.Subscript(value=GLOBAL_ARRAY, slice=ast.Index(value=ast.Constant(value=arg_id)),
                              ctx=ast.Load))

        if len(targets) > 1:
            temp = (ast.Assign(targets=[ast.Tuple(elts=targets)],
                               value=ast.Call(func=ast.Name(id=external_function_name, ctx=ast.Load()),
                                              args=args, keywords=[]), lineno=0))
        else:
            temp = (ast.Assign(targets=[targets[0]],
                               value=ast.Call(func=ast.Name(id=external_function_name, ctx=ast.Load()),
                                              args=args, keywords=[]), lineno=0))
        return temp

    def add_conditional_call(self,external_function_name, input_args, target_ids, tag, active=True):
        active=active

        arg_ids = map(lambda arg: self.variable_names[arg], input_args)
        targets=[]
        args=[]

        for target_id in target_ids:
            targets.append(
                ast.Subscript(value=GLOBAL_ARRAY,
                              slice=ast.Index(value=ast.Constant(value=self.variable_names[input_args[target_id]])),
                              ctx=ast.Store()))

        for arg_id in arg_ids:
            args.append(
                ast.Subscript(value=GLOBAL_ARRAY, slice=ast.Index(value=ast.Constant(value=arg_id)),
                              ctx=ast.Load))

        if len(targets) > 1:
            targets=[ast.Tuple(elts=targets)]
        else:
            targets=[targets[0]]

        ast_condition = ast.Name(id=external_function_name + '_flag', ctx=ast.Load())

        temp=ast.If(test=ast_condition, body=[
            ast.Assign(targets=targets, value=
                ast.Call(func=ast.Name(id=external_function_name, ctx=ast.Load()),
                                               args=args, keywords=[]),
                lineno=0)],
            orelse=[])



        setattr(temp, 'cnd', True)
        setattr(temp, 'active', active)
        self.cnd_calls.update({tag: len(self.body)})
        self.body.append(temp)

    def get_call_enabled(self, tag):
        return self.body[self.cnd_calls[tag]].active

    def set_call_enabled(self, tag, enabled):
        self.body[self.cnd_calls[tag]].active=enabled

    def add_call(self, external_function_name, input_args, target_ids):
        temp = self._create_assignments(external_function_name,input_args,target_ids)
        setattr(temp, 'cnd', False)
        self.body.append(temp)

    def add_mapping(self, args, targets):
        for target in targets:
            target = [target]
            if len(target) > 1:
                raise ValueError("Only mapping to single target is supported")
            arg_idxs = []
            for arg in args:
                arg_idxs.append(self.variable_names[arg])
            target_idx = self.variable_names[target[0]]
            print(target_idx)
            if len(args) == 1:
                temp=(ast.Assign(targets=[ast.Subscript(value=GLOBAL_ARRAY,
                                                                   slice=ast.Index(
                                                                       value=ast.Constant(value=target_idx,
                                                                                          kind=None)))],
                                            value=ast.Subscript(value=GLOBAL_ARRAY,
                                                                slice=ast.Index(
                                                                    value=ast.Constant(value=arg_idxs[0], kind=None)))
                                            , lineno=0))
            else:
                temp=(ast.Assign(targets=[ast.Subscript(value=GLOBAL_ARRAY,
                                                                   slice=ast.Index(
                                                                       value=ast.Constant(value=target_idx,
                                                                                          kind=None)))],
                                            value=ast.BinOp(left=self._generate_sum_left(arg_idxs[1:]), op=ast.Add(),
                                                            right=ast.Subscript(value=GLOBAL_ARRAY,
                                                                                slice=ast.Index(
                                                                                    value=ast.Constant(
                                                                                        value=arg_idxs[0],
                                                                                        kind=None))))
                                            , lineno=0))
            setattr(temp, 'cnd', False)

            self.body.append(temp)

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
        temp=(ast.For(target=ast.Name(id='i', ctx=ast.Store()),
                                 iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=
                                 [ast.Constant(value=len(variable_name_arg_and_trg))], keywords=[]),
                                 body=[self._generate_set_call_body(external_function_name,
                                                                    len(variable_name_arg_and_trg[0]),
                                                                    self.variable_names[
                                                                        variable_name_arg_and_trg[0][0]],
                                                                    targets_ids)],
                                 orelse=[], lineno=0))
        setattr(temp, 'cnd', False)
        self.body.append(temp)
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
                                             args=args, keywords=[]), lineno=0)
        else:
            return ast.Assign(targets=[targets[0]],
                              value=ast.Call(func=ast.Name(id=external_function_name, ctx=ast.Load()),
                                             args=args, keywords=[]), lineno=0)

    def add_set_mapping(self, args2d, targets2d):
        pass
