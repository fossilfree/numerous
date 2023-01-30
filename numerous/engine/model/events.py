import ast
import inspect

import numpy.typing as npt
import numpy as np
from numba.core.registry import CPUDispatcher
from typing import List, Callable, Tuple

from numerous.engine.model.utils import njit_and_compile_function
from numerous.engine.numerous_event import NumerousEvent


def generate_event_condition_ast(event_functions: list[NumerousEvent],
                                 from_imports: list[tuple[str, str]]) -> tuple[CPUDispatcher, npt.ArrayLike]:
    array_label = "result"
    directions_array = []
    body = [ast.Assign(targets=[ast.Name(id=array_label)], lineno=0,
                       value=ast.List(elts=[], ctx=ast.Load()))]
    compiled_functions = {}
    for event in event_functions:
        if event.compiled_functions:
            compiled_functions.update(event.compiled_functions)
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

    return njit_and_compile_function(body_r, from_imports, compiled_functions=compiled_functions), np.array(
        directions_array, dtype=np.float)


class VariablesVisitor(ast.NodeVisitor):
    def __init__(self, path_to_root_str, model, idx_type):
        self.path_to_root_str = path_to_root_str
        self.model = model
        self.idx_type = idx_type

    def generic_visit(self, node):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                for (var_path, var) in self.model.path_to_variable.items():
                    if var_path.startswith(self.path_to_root_str):
                        var_path = var_path[len(self.path_to_root_str):]
                    if var_path == node.slice.value:
                        node.slice.value = self.model._get_var_idx(var, self.idx_type)[0]
            if isinstance(node.slice.value, str):
                raise KeyError(f'No such variable: {node.slice.value}')

        ast.NodeVisitor.generic_visit(self, node)


def _replace_path_strings(model, function, idx_type, path_to_root=[]):
    if hasattr(function, 'lines'):
        lines = function.lines
    else:
        lines = inspect.getsource(function)
    path_to_root_str = ".".join(path_to_root) + "."
    func = ast.parse(lines.strip()).body[0]
    VariablesVisitor(path_to_root_str, model, idx_type).generic_visit(func)
    return func


def generate_event_action_ast(event_functions: list[NumerousEvent],
                              from_imports: list[tuple[str, str]]) -> Tuple[CPUDispatcher, List[Callable]]:
    body = []
    compiled_functions = {}
    external_functions = []
    for idx, event in enumerate(event_functions):
        if event.compiled_functions:
            compiled_functions.update(event.compiled_functions)
        if event.is_external:
            external_functions.append(event.action)
            continue
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

    internal_functions = njit_and_compile_function(body_r, from_imports, compiled_functions=compiled_functions)
    return internal_functions, external_functions
