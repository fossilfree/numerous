import ast

from numerous.engine.model.utils import njit_and_compile_function


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


def generate_event_action_ast(event_functions, from_imports):
    body = []

    for idx, (_, _, action_fun) in enumerate(event_functions):
        body.append(action_fun)
        body.append(ast.If(test=ast.Compare(left=ast.Name(id='a_idx', ctx=ast.Load()), ops=[ast.Eq()],
                                            comparators=[ast.Constant(value=idx)]),
                           body=[ast.Expr(value=ast.Call(func=ast.Name(id=action_fun.name, ctx=ast.Load()),
                                                         args=[ast.Name(id='t', ctx=ast.Load()),
                                                               ast.Name(id='variables', ctx=ast.Load())],
                                                         keywords=[]))], orelse=[]))

    body_r = ast.FunctionDef(name='action_list',
                             args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='states'),
                                                                      ast.arg(arg='a_idx')],
                                                kwonlyargs=[], kw_defaults=[], defaults=[]),
                             body=body, decorator_list=[], lineno=0)

    return njit_and_compile_function(body_r, from_imports)
