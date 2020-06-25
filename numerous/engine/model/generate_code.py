import ast, astor
from numerous.engine.model.graph import Graph
from numerous.engine.model.parser_ast import NodeTypes, EquationNode, EquationEdge, attr_ast
class dot_dict:
    def __init__(self,**d):

        for k, v in d.items():
            setattr(self, k, v)

def node_to_ast(n: EquationNode):
    if n[1].ast_type == ast.Attribute:
        return attr_ast(n[0])


def generate_code(g: Graph, func_name='kernel'):
    mod = ast.Module()
    f = ast.FunctionDef(func_name)
    f.body = []
    f.decorator_list = ['njit']
    f.args = dot_dict(args=[ast.Name(id='var')], vararg=None, defaults=[], kwarg=None)

    lineno_count = 1

    for id, n in g.nodes_map.items():
        lineno_count+=1


        if n[1].ast_type == ast.Assign:
            #n[1].id = n[0]
            print('value: ', g.edges_end(n, label='value'))
            print(n[0])
            value_node = g.nodes_map[g.edges_end(n, label='value')[0][0]]
            print(value_node[0])
            value_ast = node_to_ast(value_node)

            target_node = g.nodes_map[g.edges_start(n, label='target0')[0][1]]
            target_ast = node_to_ast(target_node)

            if value_ast and target_ast:
                ast_assign = ast.Assign(targets=[target_ast], value=value_ast)
                f.body.append(ast_assign)


            #f.body.append(assign)

    mod.body = []
    mod.body.append(f)

    source = astor.to_source(mod, indent_with=' ' * 4, add_line_information=False,
                             source_generator_class=astor.SourceGenerator)

    print(source)