from numerous.engine.model.graph import Graph
from numerous.engine.model.utils import NodeTypes, recurse_Attribute, wrap_function, dot_dict, generate_code_file
import ast
from numba import njit

def library_function(funcs_map):
    args=ast.Assign(targets=[ast.Name(id='args')], value=ast.Subscript(slice=ast.Index(value=ast.Name(id='arg_indcs')), value=ast.Name(id='variables')))

    prev = None
    funcs = list(funcs_map)
    i=0
    for i, f in enumerate(funcs):

        expr = ast.Expr(value=ast.Call(args=[ast.Name(id='*args')], func=ast.Name(id=f), keywords={}))
        ifexp = ast.If(body=[expr], orelse=[],
                       test=ast.Compare(comparators=[ast.Num(n=i)], left=ast.Name(id='index'), ops=[ast.Eq()]))

        if not prev:

            out = ifexp

        else:
            prev.orelse.append(ifexp)

        prev = ifexp

    prev.orelse.append(ast.Raise(type=ast.Call(args=[ast.Str(s='Index out of bounds')], func=ast.Name(id='IndexError'), keywords={}), inst=None, tback=None))

    body=[args, out]



    args = dot_dict(args=[ast.Name(id='index'), ast.Name(id='variables'), ast.Name(id='arg_indcs')], vararg=None, defaults=[], kwarg=None)
    decorators = [ast.Call(func=ast.Name(id='njit'), args=[ast.Str(s='void(int64, float64[:], int64[:])')], keywords={})]
    return wrap_function('library', body, args,  decorators)


preamble = """
import numpy as np
from numba import njit


@njit
def func_call(func):
    def inner(variables, args, target):
        variables[target] = func(*variables[args])
    return inner

"""


def generate_program(graph: Graph):
    nodes = graph.topological_nodes()
    ops = {}
    program = []
    for n in nodes:
        node = n[1]
        if node.node_type == NodeTypes.OP:
            if node.ast_type == ast.Call:
                this_op = recurse_Attribute(node.func)
            elif node.ast_type == ast.BinOp:
                this_op = type(node.ast_op)
            elif node.ast_type == ast.UnaryOp:
                this_op = type(node.ast_op)

            if this_op in ops:
                ix = list(ops.keys()).index(this_op)
            else:
                ix = len(ops.values())
                ops[this_op] = {}


            program.append((ix,))

    print(ops)
    body = [library_function(ops)]

    generate_code_file(body, 'libfile.py', preamble=preamble)