import ast
import inspect

import numpy.typing as npt
import numpy as np
from numba.core.registry import CPUDispatcher

from numerous.engine.model.utils import njit_and_compile_function
from numerous.engine.numerous_event import NumerousEvent


def generate_event_condition_ast(event_functions: list[NumerousEvent],
                                 from_imports: list[tuple[str, str]]) -> tuple[list[CPUDispatcher], npt.ArrayLike]:
    array_label = "result"
    directions_array = []
    body = [ast.Assign(targets=[ast.Name(id=array_label)], lineno=0,
                       value=ast.List(elts=[], ctx=ast.Load()))]

    for event in event_functions:
        directions_array.append(event.direction)
        body.append(event.condition)
        body.append(ast.Expr(value=ast.Call(
            func=ast.Attribute(value=ast.Name(id=array_label, ctx=ast.Load()), attr='append', ctx=ast.Load()),
            args=[ast.Call(func=ast.Name(id=event.condition.name, ctx=ast.Load()),
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

    return [njit_and_compile_function(body_r, from_imports)], np.array(directions_array)


def _replace_path_strings(model, function, idx_type, path_to_root=[]):
    lines = inspect.getsource(function)
    path_to_root_str = ".".join(path_to_root) + "."
    path_to_root_str_len = len(path_to_root_str)
    for (var_path, var) in model.path_to_variable.items():
        if var_path.startswith(path_to_root_str):
            var_path = var_path[path_to_root_str_len:]
        if var_path in lines:
            lines = lines.replace('[\'' + var_path + '\']', str(model._get_var_idx(var, idx_type)))
    func = ast.parse(lines.strip()).body[0]
    return func


def generate_event_action_ast(event_functions: list[NumerousEvent],
                              from_imports: list[tuple[str, str]]) -> CPUDispatcher:
    body = []

    for idx, event in enumerate(event_functions):
        body.append(event.action)
        body.append(ast.If(test=ast.Compare(left=ast.Name(id='a_idx', ctx=ast.Load()), ops=[ast.Eq()],
                                            comparators=[ast.Constant(value=idx)]),
                           body=[ast.Expr(value=ast.Call(func=ast.Name(id=event.action.name, ctx=ast.Load()),
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
