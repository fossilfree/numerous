import ast

import numpy as np
from numba.core.registry import CPUDispatcher

from numerous.engine.model.utils import njit_and_compile_function


def generate_event_condition_ast(event_functions, from_imports):
    array_label = "result"
    directions_array = []
    body = [ast.Assign(targets=[ast.Name(id=array_label)], lineno=0,
                       value=ast.List(elts=[], ctx=ast.Load()))]

    for _, cond_fun, _ in event_functions:
        directions_array.append(cond_fun.direction)
        body.append(cond_fun)
        body.append(ast.Expr(value=ast.Call(
            func=ast.Attribute(value=ast.Name(id=array_label, ctx=ast.Load()), attr='append', ctx=ast.Load()),
            args=[ast.Call(func=ast.Name(id=cond_fun.name, ctx=ast.Load()),
                           args=[ast.Name(id='t', ctx=ast.Load()), ast.Name(id='states', ctx=ast.Load())],
                           keywords=[])], keywords=[])))

    body.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='np',
                                                                            ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()),
                                          args=[ast.Name(id=array_label, ctx=ast.Load()),
                                                ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                                              ctx=ast.Load())],
                                          keywords=[])))
    body_r = ast.FunctionDef(name='condition_list',
                             args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='states')],
                                                kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=body, decorator_list=[], lineno=0)

    return njit_and_compile_function(body_r, from_imports), np.array(directions_array)


def generate_event_action_ast(event_functions: list[tuple[str, ast.FunctionDef, ast.FunctionDef]],
                              from_imports: list[tuple[str, str]]) -> CPUDispatcher:
    body = []

    for idx, (_, _, action_fun) in enumerate(event_functions):
        body.append(action_fun)
        body.append(ast.If(test=ast.Compare(left=ast.Name(id='a_idx', ctx=ast.Load()), ops=[ast.Eq()],
                                            comparators=[ast.Constant(value=idx)]),
                           body=[ast.Expr(value=ast.Call(func=ast.Name(id=action_fun.name, ctx=ast.Load()),
                                                         args=[ast.Name(id='t', ctx=ast.Load()),
                                                               ast.Name(id='states', ctx=ast.Load())],
                                                         keywords=[]))], orelse=[]))

    body.append(ast.Return(value=ast.Name(id='states', ctx=ast.Load())))

    body_r = ast.FunctionDef(name='action_list',
                             args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='states'),
                                                                      ast.arg(arg='a_idx')],
                                                kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=body, decorator_list=[], lineno=0)

    return njit_and_compile_function(body_r, from_imports)
