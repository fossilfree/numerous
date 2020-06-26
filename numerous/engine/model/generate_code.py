import ast, astor
from numerous.engine.model.graph import Graph
from numerous.engine.model.parser_ast import NodeTypes, EquationNode, EquationEdge, attr_ast
class dot_dict:
    def __init__(self,**d):

        for k, v in d.items():
            setattr(self, k, v)

def node_to_ast(n: EquationNode, g: Graph):
    if n[1].ast_type == ast.Attribute:
        return attr_ast(n[0])
    elif n[1].ast_type == ast.Name:
        return attr_ast(n[0])
    elif n[1].ast_type == ast.Num:
        return attr_ast(n[0])
    elif n[1].ast_type == ast.BinOp:

        left_node = g.nodes_map[g.edges_end(n, label='left')[0][0]]
        left_ast = node_to_ast(left_node, g)

        right_node = g.nodes_map[g.edges_end(n, label='right')[0][0]]
        right_ast = node_to_ast(right_node, g)

        ast_binop = ast.BinOp(left=left_ast, right=right_ast, op=n[1].ast_op)
        return ast_binop

    elif n[1].ast_type == ast.UnaryOp:
        operand = g.nodes_map[g.edges_end(n, label='operand')[0][0]]
        operand_ast = node_to_ast(operand, g)

        ast_unop = ast.UnaryOp(operand=operand_ast, op=n[1].ast_op)
        return ast_unop

    elif n[1].ast_type == ast.Call:

        args = [g.nodes_map[ii[0]] for ii in g.edges_end(n, label='args')]
        args_ast = []
        for a in args:
            a_ast = node_to_ast(a, g)
            args_ast.append(a_ast)

        ast_Call = ast.Call(args=args_ast, func=n[1].func, keywords={})

        return ast_Call

    #TODO implement missing code ast objects
    raise TypeError(f'Cannot convert {n[1]},{n[1].ast_type}')


def generate_code(g: Graph, var_map, indcs, func_name='kernel'):
    mod = ast.Module()
    mod.body = []
    #mod.body.append(ast.ImportFrom(level=0, module='numba', names=[ast.(alias=ast.Dict(asname=None, name='njit'))]))
    """
    mod.body.append(ast.Assign(
        targets=[ast.Name(id='states_ix')], value=ast.Call(args=[ast.Num(n=i) for i in indcs[0]], func=ast.Name(id='slice'), keywords={})))
    mod.body.append(ast.Assign(
        targets=[ast.Name(id='deriv_ix')],
        value=ast.Call(args=[ast.Num(n=i) for i in indcs[1]], func=ast.Name(id='slice'), keywords={})))
    mod.body.append(ast.Assign(
        targets=[ast.Name(id='mapping_ix')],
        value=ast.Call(args=[ast.Num(n=i) for i in indcs[2]], func=ast.Name(id='slice'), keywords={})))
    """
    """
    mod.body.append(ast.Assign(
        targets=[ast.Name(id='states_ix')],
        value=ast.Tuple(elts=[ast.Num(n=i) for i in indcs[0]])))
    mod.body.append(ast.Assign(
        targets=[ast.Name(id='deriv_ix')],
        value=ast.Tuple(elts=[ast.Num(n=i) for i in indcs[1]])))
    mod.body.append(ast.Assign(
        targets=[ast.Name(id='mapping_ix')],
        value=ast.Tuple(elts=[ast.Num(n=i) for i in indcs[2]])))
    """
    f = ast.FunctionDef(func_name)
    f.body = []
    f.body.append(ast.Assign(
        targets=[ast.Subscript(slice=ast.Slice(lower=ast.Num(n=indcs[0][0]),upper=ast.Num(n=indcs[0][1]),step=None), value=ast.Name(id='variables'))],
        value=ast.Name(id='y')))
    f.body.append(ast.Assign(
        targets=[ast.Subscript(slice=ast.Slice(lower=ast.Num(n=indcs[1][0]),upper=ast.Num(n=indcs[1][1]),step=None), value=ast.Name(id='variables')),
                 ast.Subscript(slice=ast.Slice(lower=ast.Num(n=indcs[2][0]),upper=ast.Num(n=indcs[2][1]), step=None), value=ast.Name(id='variables'))
                 ],
        value=ast.Num(n=0)))


    #f.decorator_list = [ast.Call(func=ast.Name(id='njit'), keywords=[], args=[ast.Str(s='f8[:](f8[:], i4[:])')])]
    f.decorator_list = [ast.Call(func=ast.Name(id='njit'), keywords=[], args=[ast.Str(s='f8[:](f8[:], f8[:])')])]

    f.args = dot_dict(args=[ast.Name(id='variables'), ast.Name(id='y')], vararg=None, defaults=[], kwarg=None)

    f.body.append(ast.Assign(targets=[ast.Tuple(elts=[attr_ast(v) for v in var_map])],value=ast.Name(id='variables')))

    lineno_count = 1
    top_nodes = g.topological_nodes()
    #top_nodes=[]
    print('creating ast')
    for n in top_nodes:
        lineno_count+=1

        if n[1].ast_type == ast.Assign or n[1].ast_type == ast.AugAssign:
            #n[1].id = n[0]
            value_node = g.nodes_map[g.edges_end(n, label='value')[0][0]]
            value_ast = node_to_ast(value_node, g)

            target_node = g.nodes_map[g.edges_start(n, label='target0')[0][1]]
            target_ast = node_to_ast(target_node, g)

            if value_ast and target_ast:
                if n[1].ast_type == ast.Assign:
                    ast_assign = ast.Assign(targets=[target_ast], value=value_ast)
                else:
                    ast_assign = ast.AugAssign(target=target_ast, value=value_ast, op=ast.Add())
                f.body.append(ast_assign)


            #f.body.append(assign)
    f.body.append(ast.Assign(targets=[ast.Subscript(slice=ast.Slice(lower=None, upper=None, step=None), value=ast.Name(id='variables'))], value=ast.Tuple(elts=[attr_ast(v) for v in var_map])))

    f.body.append(ast.Return(
        value=ast.Subscript(slice=ast.Slice(lower=ast.Num(n=indcs[1][0]),upper=ast.Num(n=indcs[1][1]), step=None), value=ast.Name(id='variables'))
    ))

    mod.body.append(f)
    print('Code generation')

    source = "from numba import njit\nimport numpy as np\n"+astor.to_source(mod, indent_with=' ' * 4, add_line_information=False,
                             source_generator_class=astor.SourceGenerator)

    if len(source)<10000:
        print('Source code')
        print(source)

    print('Compiling')
    import timeit
    print('Compile time: ', timeit.timeit(
            lambda: exec(source, globals()),  number=1))

    return kernel
    #from numba.pycc import CC

    #cc = CC('my_module')

    #cc.export('multf', 'f8[:](f8[:], i4[:])')(kernel)

    #print('cc')
    #print('Compile time: ', timeit.timeit(
    #    lambda: cc.compile(), number=1))

    #print('cc done')
    #Dry run
    #print('Dry run')
    #variables = np.zeros()
    #print('Dry run time: ', timeit.timeit(
    #    lambda: kernel(variables, y), number=1))