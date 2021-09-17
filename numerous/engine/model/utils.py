import ast
from enum import unique, IntEnum


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


def recurse_Attribute(attr, sep='.'):
    if hasattr(attr, 'id'):
        return attr.id
    elif isinstance(attr.value, ast.Name):
        return attr.value.id + sep + attr.attr
    elif isinstance(attr.value, ast.Attribute):
        return recurse_Attribute(attr.value) + sep + attr.attr


def njit_and_compile_function(func, from_imports):
    fname = func.name
    njit_decorator = ast.Name(id='njit', ctx=ast.Load())
    func.decorator_list = [njit_decorator]
    body = []
    for (module, label) in from_imports:
        body.append(
            ast.ImportFrom(module=module, names=[ast.alias(name=label, asname=None)], lineno=0, col_offset=0,
                           level=0))
    body.append(func)
    body.append(
        ast.Return(value=ast.Name(id=fname, ctx=ast.Load(), lineno=0, col_offset=0), lineno=0, col_offset=0))

    func = wrap_function(fname + '1', body, decorators=[],
                         args=ast.arguments(posonlyargs=[], args=[], vararg=None, defaults=[],
                                            kwonlyargs=[], kw_defaults=[], lineno=0, kwarg=None))
    module_func = ast.Module(body=[func], type_ignores=[])
    code = compile(ast.parse(ast.unparse(module_func)), filename='event_storage', mode='exec')
    namespace = {}
    exec(code, namespace)
    compiled_func = list(namespace.values())[1]()
    return compiled_func


def generate_event_action_ast(event_functions, from_imports):
    # if a_idx == 888: ac_fun(t, variables)
    body = []

    for idx, (_, _, action_fun) in enumerate(event_functions):
        body.append(action_fun)
        body.append(ast.If(test=ast.Compare(left=ast.Name(id='a_idx', ctx=ast.Load()), ops=[ast.Eq()],
                                        comparators=[ast.Constant(value=idx)]),
                           body=[ast.Expr(value=ast.Call(func=ast.Name(id=action_fun.name, ctx=ast.Load()),
                                                 args=[ast.Name(id='t', ctx=ast.Load()), ast.Name(id='variables', ctx=ast.Load())],
                                                 keywords=[]))], orelse=[]))

    body_r = ast.FunctionDef(name='action_list',
                             args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='states')],
                                                kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=body, decorator_list=[], lineno=0)

    return njit_and_compile_function(body_r, from_imports)


def generate_event_condition_ast(event_functions, from_imports):
    body = [ast.Assign(targets=[ast.Name(id="results")], lineno=0,
                       value=ast.List(elts=[], ctx=ast.Load()))]

    for _, cond_fun, _ in event_functions:
        body.append(cond_fun)
        body.append(ast.Expr(value=ast.Call(
            func=ast.Attribute(value=ast.Name(id='result', ctx=ast.Load()), attr='append', ctx=ast.Load()),
            args=[ast.Call(func=ast.Name(id=cond_fun.name, ctx=ast.Load()),
                           args=[ast.Name(id='t', ctx=ast.Load()), ast.Name(id='states', ctx=ast.Load())],
                           keywords=[])], keywords=[])))

    body.append(ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='np',
                                                                            ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()),
                                          args=[ast.Name(id='result', ctx=ast.Load()),
                                                ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                                              ctx=ast.Load())],
                                          keywords=[])))
    body_r = ast.FunctionDef(name='condition_list',
                             args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='states')],
                                                kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=body, decorator_list=[], lineno=0)

    return njit_and_compile_function(body_r, from_imports)


def wrap_function(name, body, args, decorators):
    f = ast.FunctionDef(name)
    f.body = body
    f.decorator_list = decorators
    f.args = args
    f.lineno = 0
    f.col_offset = 0
    return f
