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
    out=None
    for i, f in enumerate(funcs_map.items()):

        if f[1]['op'] == 'summing':
            expr = ast.Assign(
                targets=[ast.Subscript(slice=ast.Name(id='target_indcs'), value=ast.Name(id='variables'))],
                value=ast.Call(
                    args=[ast.Name(id='args')], func=ast.Name(id='np.sum'), keywords={}))


        elif f[1]['op_type'] == ast.Call:

            #print(f[0])
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
    if prev:
        prev.orelse.append(ast.Raise(type=ast.Call(args=[ast.Str(s='Index out of bounds')], func=ast.Name(id='IndexError'), keywords={}), inst=None, tback=None))

    body=[args]

    if out:
        body+=[out]



    args = dot_dict(args=[ast.Name(id='index'), ast.Name(id='variables'), ast.Name(id='arg_indcs'),  ast.Name(id='target_indcs')], vararg=None, defaults=[], kwarg=None)
    decorators = [ast.Call(func=ast.Name(id='njit'), args=[ast.Str(s='void(int64, float64[:], int64[:], int64[:])')], keywords={})]
    return wrap_function('library', body, args,  decorators)


preamble = """
import numpy as np
from numba import njit, float64
"""

#TODO lookup var indices!

def generate_program(graph: Graph, variables, indcs):
    print('variables: ', variables)
    nodes = graph.topological_nodes()
    ops = {}
    program = []
    llvm_program = []
    llvm_funcs = []
    indices = []
    for n in nodes:
        #print(n[0])

        node = n
        #print(node.node_type)
        print('n: ',n)
        start_arg = len(indices)
        nt = graph.get(n, 'node_type')
        ast_type = graph.get(n, 'ast_type')
        if nt == NodeTypes.OP or nt == NodeTypes.ASSIGN or nt == NodeTypes.SUM or nt == NodeTypes.EQUATION:

            if nt == NodeTypes.SUM:

                this_op = 'summing'
                this_op_type = 'summing'
                graph.nodes_attr['ast_type'][node] = ast.Call
                #(this_op)
                args = [graph.key_map[ii[0]] for ii in graph.get_edges_for_node_filter(end_node=n, attr='e_type', val='value')[1]]

                targets = [graph.key_map[ii[1]] for ii in
                        graph.get_edges_for_node_filter(start_node=n, attr='e_type', val='target')[1]]
                #('targets:', targets)
                lenargs = len(args)
                lentargets = len(targets)
                indices += [variables.index(a) for a in args+targets]

                llvm_program.append({'func': 'sum', 'targets': [t for t in targets], 'args': [a for a in args]})


            elif ast_type == ast.Call and nt == NodeTypes.EQUATION:
                print(n)
                print(graph.key_map[n])



                this_op = this_op_type = recurse_Attribute(graph.get(n,'func'))
                #print(this_op)
                # TODO this is really a hack!
                #args = node.scope_var['args']
                args = graph.get(n, 'scope_var')['args']
                targets = graph.get(n, 'scope_var')['targets']
                #targets = node.scope_var['targets']
                #print(args)

                lenargs = len(args)
                lentargets = len(targets)
                #('targets:', targets)
                indices+=[variables.index(a) for a in args+targets]

                llvm_program.append({'func': 'call', 'ext_func': this_op_type, 'args': args, 'targets': targets})

            elif ast_type == ast.Call:

                this_op = this_op_type = recurse_Attribute(node.func)
                #print(this_op)
                args = [graph.key_map[ii] for ii in
                        graph.get_edges_for_node_filter(end_node=n, attr='e_type', val='value')[1]]

                targets = [graph.key_map[ii] for ii in
                           graph.get_edges_for_node_filter(start_node=n, attr='e_type', val='target')[1]]
                lenargs = len(args)
                lentargets = len(targets)
                #('targets:', targets)
                indices += [variables.index(a[0]) for a in args+targets]
                llvm_program.append({'func': 'call', 'ext_func': this_op_type, 'args': args, 'targets': targets})
            """
            elif node.ast_type == ast.BinOp:

                this_op = node.ast_op
                this_op_type = type(this_op)
                lenargs = 2
                lentargets = 1
                indices.append([variables.index(a[0]) for a in args])

            elif node.ast_type == ast.UnaryOp:

                this_op = node.ast_op
                this_op_type = type(this_op)
                lenargs = 1
                
                indices.append([variables.index(a[0]) for a in args])

            elif node.ast_type == ast.Assign:

                this_op = ast.Assign
                this_op_type = type(this_op)
                lenargs = 1
                indices.append([variables.index(a[0]) for a in args])

            elif node.ast_type == ast.AugAssign:
                print(n)
                this_op = node.ast_op
                this_op_type = type(this_op)
                lenargs = 1
                indices.append([variables.index(a[0]) for a in args])
            """
            if this_op_type in ops:
                ix = list(ops.keys()).index(this_op_type)
            else:
                ix = len(ops.values())
                ops[this_op_type] = dict(lenargs=lenargs, lentargets=lentargets, op_type=graph.get(node, 'ast_type'), op=this_op)
            #print(lentargets)
            end_arg = start_arg + lenargs
            end_targets = end_arg + lentargets

            program.append((ix,start_arg, end_arg, end_targets))



    body = [library_function(ops)]

    #source = generate_code_file(body, 'libfile.py', preamble=preamble)

    args = dot_dict(args=[ast.Name(id='program'), ast.Name(id='variables'), ast.Name(id='indices')], vararg=None, defaults=[], kwarg=None)
    program_body = [
        ast.For(body=[ast.Expr(value=ast.Call(
            args=[
                ast.Subscript(slice=ast.Index(value=ast.Num(n=0)), value=ast.Name(id='p')),
                ast.Name(id='variables'),
                ast.Subscript(slice=ast.Slice(
                    lower=ast.Subscript(slice=ast.Index(value=ast.Num(n=1)), value=ast.Name(id='p')),
                    upper=ast.Subscript(slice=ast.Index(value=ast.Num(n=2)), value=ast.Name(id='p')),
                    step=None
                ), value=ast.Name(id='indices')),
                ast.Subscript(slice=ast.Slice(
                    lower=ast.Subscript(slice=ast.Index(value=ast.Num(n=2)), value=ast.Name(id='p')),
                    upper=ast.Subscript(slice=ast.Index(value=ast.Num(n=3)), value=ast.Name(id='p')),
                    step=None
                ), value=ast.Name(id='indices')),
            ],
            func=ast.Name(id='library'),
            keywords=[]
        )
        )],
        iter=ast.Name(id='program'),
        orelse= [],
        target=ast.Name(id='p')

        )
    ]
    run_program_ast = wrap_function(name='run_program', args=args, body=program_body, decorators=["njit('void(int64[:,:], float64[:], int64[:])')"])
    body.append(run_program_ast)

    args = dot_dict(args=[ast.Name(id='variables'), ast.Name(id='y'), ast.Name(id='program'), ast.Name(id='indices')], vararg=None,
                    defaults=[], kwarg=None)
    diff_body = [
        ast.Assign(targets=[ast.Subscript(slice=ast.Slice(lower=ast.Num(n=0), upper=ast.Num(n=indcs[0]), step=None), value=ast.Name(id='variables'))], value=ast.Name(id='y')),
        ast.Expr(value=ast.Call(args=[ast.Name(id='program'), ast.Name(id='variables'), ast.Name(id='indices')], func=ast.Name(id='run_program'), keywords=[])),
        ast.Return(value=ast.Subscript(slice=ast.Slice(lower=ast.Num(n=indcs[1]), upper=ast.Num(n=indcs[1]+indcs[2]), step=None), value=ast.Name(id='variables')))
    ]
    diff_ast = wrap_function(name='diff', args=args, body=diff_body,
                                    decorators=["njit('float64[:](float64[:], float64[:], int64[:,:], int64[:])')"])

    body.append(diff_ast)


    run_program_source = """
    @njit('void(int64[:,:], float64[:], int64[:])')
    def run_program(program, variables, indices):
        for p in program:
            library(p[0], variables, indices[p[1]:p[2]], indices[p[2]])
    """

    return run_program_source, body, program, indices, llvm_program
