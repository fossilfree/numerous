import ast
from model.grpah_representation.graph import Graph
from numerous.engine.model.parser_ast import EquationNode


class dot_dict:
    def __init__(self, **d):
        for k, v in d.items():
            setattr(self, k, v)


def node_to_ast(n: EquationNode, g: Graph, var_def):
    if n[1].ast_type == ast.Attribute:
        return var_def(n[0])
    elif n[1].ast_type == ast.Name:
        return var_def(n[0])
    elif n[1].ast_type == ast.Num:
        return ast.Call(args=[ast.Num(value=n[1].value)], func=ast.Name(id='float64'), keywords={})
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

    elif n[1].ast_type == ast.IfExp:
        body = g.nodes_map[g.edges_end(n, label='body')[0][0]]
        body_ast = node_to_ast(body, g)

        orelse = g.nodes_map[g.edges_end(n, label='orelse')[0][0]]
        orelse_ast = node_to_ast(orelse, g)

        test = g.nodes_map[g.edges_end(n, label='test')[0][0]]
        test_ast = node_to_ast(test, g)

        ast_ifexp = ast.IfExp(body=body_ast, orelse=orelse_ast, test=test_ast)

        return ast_ifexp

    elif n[1].ast_type == ast.Compare:
        comp = [g.nodes_map[ii[0]] for ii in g.edges_end(n, label='comp')]
        comp_ast = []
        for a in comp:
            a_ast = node_to_ast(a, g)
            comp_ast.append(a_ast)

        left = g.nodes_map[g.edges_end(n, label='left')[0][0]]
        left_ast = node_to_ast(left, g)

        ast_Comp = ast.Compare(left=left_ast, comparators=comp_ast, ops=n[1].ops)

        return ast_Comp

    # TODO implement missing code ast objects
    raise TypeError(f'Cannot convert {n[1]},{n[1].ast_type}')


def wrap_module(body):
    mod = ast.Module()
    mod.body = body
    return mod


def wrap_function(name, body, args, decorators):
    f = ast.FunctionDef(name)
    f.body = body
    f.decorator_list = decorators
    f.args = args
    return f


class Vardef:
    def __init__(self):
        self.vars_inds_map = []

    def var_def(self, var):
        if not var in self.vars_inds_map:
            self.vars_inds_map.append(var)
        ix = self.vars_inds_map.index(var)

        return ast.Subscript(slice=ast.Index(value=ast.Num(n=ix)), value=ast.Name(id='s'))


def function_from_graph(g: Graph):
    lineno_count = 1
    top_nodes = g.topological_nodes()

    var_def = Vardef().var_def

    print('creating ast')

    body = []

    for n in top_nodes:
        lineno_count += 1

        if n[1].ast_type == ast.Assign or n[1].ast_type == ast.AugAssign:
            # n[1].id = n[0]
            value_node = g.nodes_map[g.edges_end(n, label='value')[0][0]]
            value_ast = node_to_ast(value_node, g, var_def)

            target_node = g.nodes_map[g.edges_start(n, label='target0')[0][1]]
            target_ast = node_to_ast(target_node, g, var_def)

            if value_ast and target_ast:
                if n[1].ast_type == ast.Assign:
                    ast_assign = ast.Assign(targets=[target_ast], value=value_ast)
                else:
                    ast_assign = ast.AugAssign(target=target_ast, value=value_ast, op=ast.Add())
                body.append(ast_assign)

    args = dot_dict(args=[ast.Name(id='l')], vararg=None, defaults=[], kwarg=None)

    decorators = [ast.Call(func=ast.Name(id='njit'), args=[ast.Str(s='(float64[:])')])]

    func = wrap_function('diff', body, decorators=decorators, args=args)

    return func

