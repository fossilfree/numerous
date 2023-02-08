import ast
import inspect

import numpy.typing as npt
import numpy as np
from numba.core.registry import CPUDispatcher
from typing import Union, Tuple

from numerous.engine.model.utils import njit_and_compile_function
from numerous.engine.numerous_event import NumerousEvent, StateEvent, TimestampEvent


def generate_event_condition_ast(event_functions: list[Union[StateEvent, TimestampEvent]],
                                 from_imports: list[tuple[str, str]]) -> tuple[CPUDispatcher, npt.ArrayLike]:
    array_label = "result"
    directions_array = []
    body = [ast.Assign(targets=[ast.Name(id=array_label)], lineno=0,
                       value=ast.List(elts=[], ctx=ast.Load()))]
    compiled_functions = {}
    closurevariables = None
    for event in event_functions:
        if event.compiled_functions:
            compiled_functions.update(event.compiled_functions)
        condition = event.condition.ast_func
        closurevariables = event.condition.closure_variables

        directions_array.append(event.direction)
        body.append(condition)
        body.append(ast.Expr(value=ast.Call(
            func=ast.Attribute(value=ast.Name(id=array_label, ctx=ast.Load()), attr='append', ctx=ast.Load()),
            args=[ast.Call(func=ast.Name(id=condition.name, ctx=ast.Load()),
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

    return njit_and_compile_function(body_r, from_imports, compiled_functions=compiled_functions,
                                     closurevariables=closurevariables), np.array(
        directions_array, dtype=np.float)


class VariablesVisitor(ast.NodeVisitor):
    def __init__(self, path_to_root_str, model, idx_type, closurevariables):
        self.path_to_root_str = path_to_root_str
        self.model = model
        self.idx_type = idx_type
        self.closurevariables = closurevariables
        self.assigned_variables = {}

    def generic_visit(self, node):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                for (var_path, var) in self.model.path_to_variable.items():
                    if var_path.startswith(self.path_to_root_str):
                        var_path = var_path[len(self.path_to_root_str):]
                    if var_path == node.slice.value:
                        node.slice.value = self.model._get_var_idx(var, self.idx_type)[0]
                        break
            if isinstance(node.slice.value, str):
                raise KeyError(f'No such variable: {node.slice.value}')
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Name):
            for (var_path, var) in self.model.path_to_variable.items():
                if var_path.startswith(self.path_to_root_str):
                    var_path = var_path[len(self.path_to_root_str):]
                if var_path in self.assigned_variables[node.slice.id]:
                    node.slice = ast.Constant(value=self.model._get_var_idx(var, self.idx_type)[0])
                    break
        if isinstance(node, ast.Name):
            if node.id in self.closurevariables:
                node = ast.Constant(value=self.closurevariables[node.id])

        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node: ast.Assign):
        evaluated = None
        if isinstance(node.value, ast.BinOp):
            operator = node.value.op
            left = node.value.left

            right = node.value.right
            left_val = self._get_value(left)
            right_val = self._get_value(right)
            if not left_val or not right_val:
                self.generic_visit(node)
                return

            if isinstance(operator, ast.Add):
                evaluated = left_val + right_val
            else:
                raise NotImplementedError(f"{type(operator)} not yet implemented")
            self.assigned_variables.update({node.targets[0].id: evaluated})
        else:
            self.generic_visit(node)

    def _get_value(self, val):
        if isinstance(val, ast.Name):
            if val.id in self.closurevariables:
                return self.closurevariables[val.id]
            else:
                return
        if isinstance(val, ast.Constant):
            return val.value

def replace_path_strings(model, function, idx_type, path_to_root=[]) -> Tuple:
    if hasattr(function, 'lines'):
        lines = function.lines
    else:
        lines = inspect.getsource(function)
    closurevariables = inspect.getclosurevars(function).nonlocals
    path_to_root_str = ".".join(path_to_root) + "."
    func = ast.parse(lines.strip()).body[0]
    VariablesVisitor(path_to_root_str, model, idx_type, closurevariables).generic_visit(func)
    return func, closurevariables


def generate_event_action_ast(event_functions: list[NumerousEvent],
                              from_imports: list[tuple[str, str]]) -> CPUDispatcher:
    body = []
    compiled_functions = {}
    closurevariables = None

    for idx, event in enumerate(event_functions):
        if event.compiled_functions:
            compiled_functions.update(event.compiled_functions)
        if event.is_external:
            continue
        action = event.action.ast_func
        closurevariables = event.action.closure_variables

        body.append(action)
        body.append(ast.If(test=ast.Compare(left=ast.Name(id='a_idx', ctx=ast.Load()), ops=[ast.Eq()],
                                            comparators=[ast.Constant(value=idx)]),
                           body=[ast.Expr(value=ast.Call(func=ast.Name(id=action.name, ctx=ast.Load()),
                                                         args=[ast.Name(id='t', ctx=ast.Load()),
                                                               ast.Name(id='states', ctx=ast.Load())],
                                                         keywords=[]))], orelse=[]))

    body.append(ast.Return(value=ast.Name(id='states', ctx=ast.Load())))

    body_r = ast.FunctionDef(name='action_list',
                             args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='states'),
                                                                      ast.arg(arg='a_idx')],
                                                kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=body, decorator_list=[], lineno=0)

    internal_functions = njit_and_compile_function(body_r, from_imports, compiled_functions=compiled_functions,
                                                   closurevariables=closurevariables)
    return internal_functions
