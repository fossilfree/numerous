import ast
from enum import unique, IntEnum

from numba.core.registry import CPUDispatcher


class Imports:
    def __init__(self):
        self.as_imports = []
        self.from_imports = []

    def add_as_import(self, module_name, alias):
        self.as_imports.append((module_name, alias))

    def add_from_import(self, module_name, element):
        self.from_imports.append((module_name, element))


@unique
class EventTypes(IntEnum):
    TIME_EVENT = 0
    STATE_EVENT = 1


# TODO add setVAR/GridVar?
@unique
class NodeTypes(IntEnum):
    OP = 0
    ASSIGN = 1
    VAR = 2
    EQUATION = 3
    SUM = 4
    TMP = 5
    ASSIGNTUPLE = 6
    IF = 7
    FMU_EQUATION = 8
    SUBSCRIPT = 9


def recurse_Attribute(attr, sep='.'):
    if hasattr(attr, 'id'):
        return attr.id
    elif isinstance(attr.value, ast.Name):
        return attr.value.id + sep + attr.attr
    elif isinstance(attr.value, ast.Attribute):
        return recurse_Attribute(attr.value) + sep + attr.attr


def njit_and_compile_function(func: ast.FunctionDef, from_imports: list[(str, str)],
                              compiled_functions: list[CPUDispatcher] = None,
                              closurevariables: dict=None) -> CPUDispatcher:
    fname = func.name
    njit_decorator = ast.Name(id='njit', ctx=ast.Load())
    func.decorator_list = [njit_decorator]
    body = []
    for (module, label) in from_imports:
        body.append(
            ast.ImportFrom(module=module, names=[ast.alias(name=label, asname=None)], lineno=0, col_offset=0,
                           level=0))
    body.append(ast.Import(names=[ast.alias(name='numpy', asname='np')]))
    body.append(func)
    body.append(
        ast.Return(value=ast.Name(id=fname, ctx=ast.Load(), lineno=0, col_offset=0), lineno=0, col_offset=0))
    wrapper_name = fname + '1'
    func = wrap_function(fname + '1', body, decorators=[],
                         args=ast.arguments(posonlyargs=[], args=[], vararg=None, defaults=[],
                                            kwonlyargs=[], kw_defaults=[], lineno=0, kwarg=None))
    module_func = ast.Module(body=[func], type_ignores=[])
    code = compile(ast.parse(ast.unparse(module_func)), filename='event_storage', mode='exec')
    if compiled_functions:
        namespace = compiled_functions
    else:
        namespace = {}

    if closurevariables:
        namespace.update(closurevariables)

    exec(code, namespace)
    compiled_func = namespace[wrapper_name]()
    return compiled_func


def wrap_function(name, body, args, decorators):
    f = ast.FunctionDef(name)
    f.body = body
    f.decorator_list = decorators
    f.args = args
    f.lineno = 0
    f.col_offset = 0
    return f
