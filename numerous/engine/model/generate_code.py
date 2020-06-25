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


    raise TypeError(f'Cannot convert {n[1]},{n[1].ast_type}')


def generate_code(g: Graph, func_name='kernel'):
    mod = ast.Module()
    f = ast.FunctionDef(func_name)
    f.body = []
    f.decorator_list = ['njit']
    f.args = dot_dict(args=[ast.Name(id='var')], vararg=None, defaults=[], kwarg=None)

    lineno_count = 1

    for n in g.topological_nodes():
        lineno_count+=1


        if n[1].ast_type == ast.Assign:
            #n[1].id = n[0]
            value_node = g.nodes_map[g.edges_end(n, label='value')[0][0]]
            value_ast = node_to_ast(value_node, g)

            target_node = g.nodes_map[g.edges_start(n, label='target0')[0][1]]
            target_ast = node_to_ast(target_node, g)

            if value_ast and target_ast:
                ast_assign = ast.Assign(targets=[target_ast], value=value_ast)
                f.body.append(ast_assign)


            #f.body.append(assign)

    mod.body = []
    mod.body.append(f)

    source = astor.to_source(mod, indent_with=' ' * 4, add_line_information=False,
                             source_generator_class=astor.SourceGenerator)

    print(source)