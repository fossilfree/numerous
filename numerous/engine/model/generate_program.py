from numerous.engine.model.graph import Graph
from numerous.engine.model.utils import NodeTypes, recurse_Attribute, wrap_function, dot_dict, generate_code_file
import ast
from numba import njit
import numpy as np

def library_function(funcs_map):
    args=ast.Assign(targets=[ast.Name(id='args')], value=ast.Subscript(slice=ast.Index(value=ast.Name(id='arg_indcs')), value=ast.Name(id='variables')))


    prev = None
    #funcs = list(funcs_map)
    i=0
    for i, f in enumerate(funcs_map.items()):
        f[1]['op']
        if f[1]['op_type'] == ast.Call:
            print(f[0])
            expr = ast.Assign(targets=[ast.Subscript(slice=ast.Name(id='target_indcs'), value=ast.Name(id='variables'))], value=ast.Call(args=[ast.Subscript(slice=ast.Index(value=ast.Num(n=i)), value=ast.Name(id='args')) for i in range(f[1]['lenargs'])], func=ast.Name(id=f[1]['op']), keywords={}))

        elif f[1]['op_type'] == ast.BinOp:
            expr = ast.Assign(targets=[ast.Subscript(slice=ast.Name(id='target_indcs'), value=ast.Name(id='variables'))], value=ast.BinOp(left=ast.Subscript(slice=ast.Index(value=ast.Num(n=0)), value=ast.Name(id='args')), right=ast.Subscript(slice=ast.Index(value=ast.Num(n=1)), value=ast.Name(id='args')), op=f[1]['op']))

        elif f[1]['op_type'] == ast.UnaryOp:
            expr = ast.Assign(targets=[ast.Subscript(slice=ast.Name(id='target_indcs'), value=ast.Name(id='variables'))], value=ast.UnaryOp(operand=ast.Subscript(slice=ast.Index(value=ast.Num(n=0)), value=ast.Name(id='args')), op=f[1]['op']))

        elif f[1]['op_type'] == ast.Assign:
            expr = ast.Assign(targets=[ast.Subscript(slice=ast.Name(id='target_indcs'), value=ast.Name(id='variables'))], value=ast.Subscript(slice=ast.Index(value=ast.Num(n=0)), value=ast.Name(id='args')))

        elif f[1]['op_type'] == ast.AugAssign:
            expr = ast.AugAssign(target=ast.Subscript(slice=ast.Name(id='target_indcs'), value=ast.Name(id='variables')), value=ast.Subscript(slice=ast.Index(value=ast.Num(n=0)), value=ast.Name(id='args')), op=f[1]['op'])


        ifexp = ast.If(body=[expr], orelse=[],
                       test=ast.Compare(comparators=[ast.Num(n=i)], left=ast.Name(id='index'), ops=[ast.Eq()]))

        if not prev:

            out = ifexp

        else:
            prev.orelse.append(ifexp)

        prev = ifexp

    prev.orelse.append(ast.Raise(type=ast.Call(args=[ast.Str(s='Index out of bounds')], func=ast.Name(id='IndexError'), keywords={}), inst=None, tback=None))

    body=[args, out]



    args = dot_dict(args=[ast.Name(id='index'), ast.Name(id='variables'), ast.Name(id='arg_indcs'),  ast.Name(id='target_indcs')], vararg=None, defaults=[], kwarg=None)
    decorators = [ast.Call(func=ast.Name(id='njit'), args=[ast.Str(s='void(int64, float64[:], int64[:], int64)')], keywords={})]
    return wrap_function('library', body, args,  decorators)


preamble = """
import numpy as np
from numba import njit, float64
"""


def generate_program(graph: Graph):
    nodes = graph.topological_nodes()
    ops = {}
    program = []
    indices = []
    for n in nodes:
        node = n[1]

        start_arg = len(indices)

        if node.node_type == NodeTypes.OP or node.node_type == NodeTypes.ASSIGN:
            if node.ast_type == ast.Call:

                this_op = this_op_type = recurse_Attribute(node.func)
                print(this_op)
                args = [graph.nodes_map[ii[0]] for ii in graph.edges_end(n, label='args')]

                lenargs = len(args)

                indices.append([0, 0, 0])

            elif node.ast_type == ast.BinOp:

                this_op = node.ast_op
                this_op_type = type(this_op)
                lenargs = 2
                indices.append([0, 0, 0])

            elif node.ast_type == ast.UnaryOp:

                this_op = node.ast_op
                this_op_type = type(this_op)
                lenargs = 1
                indices.append([0, 0])

            elif node.ast_type == ast.Assign:

                this_op = ast.Assign
                this_op_type = type(this_op)
                lenargs = 1
                indices.append([0, 0])

            elif node.ast_type == ast.AugAssign:
                print(n)
                this_op = node.ast_op
                this_op_type = type(this_op)
                lenargs = 1
                indices.append([0, 0])

            if this_op_type in ops:
                ix = list(ops.keys()).index(this_op_type)
            else:
                ix = len(ops.values())
                ops[this_op_type] = dict(lenargs=lenargs, op_type=node.ast_type, op=this_op)

            end_arg = start_arg + lenargs
            program.append((ix,start_arg, end_arg))

    print(program)
    body = [library_function(ops)]

    source = generate_code_file(body, 'libfile.py', preamble=preamble)
    #library = None
    exec(source, globals())

    library(2, np.zeros(10, np.float64), np.array([0, 0], np.int64), 0)

    @njit('void(int64[:,:], float64[:], int64[:])')
    def run_program(program, variables, indices):
        for p in program:
            library(p[0], variables, indices[p[1]:p[2]], indices[p[2]])
