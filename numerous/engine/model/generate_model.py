from numerous.engine.variables import VariableType
import ast, astor
from numerous.engine.model.graph import Graph
from numerous.engine.model.parser_ast import EquationNode, EquationEdge, attr_ast
from numerous.engine.model.utils import NodeTypes

class dot_dict:
    def __init__(self, **d):
        for k, v in d.items():
            setattr(self, k, v)


def node_to_ast(n: EquationNode, g: Graph, var_def):
    try:
        if n[1].ast_type == ast.Attribute:
            return var_def(n[0])
        elif n[1].ast_type == ast.Name:
            return var_def(n[0])
        elif n[1].ast_type == ast.Num:
            return ast.Call(args=[ast.Num(value=n[1].value)], func=ast.Name(id='float64'), keywords={})
        elif n[1].ast_type == ast.BinOp:

            left_node = g.nodes_map[g.edges_end(n, label='left')[0][0]]
            left_ast = node_to_ast(left_node, g, var_def)

            right_node = g.nodes_map[g.edges_end(n, label='right')[0][0]]
            right_ast = node_to_ast(right_node, g, var_def)

            ast_binop = ast.BinOp(left=left_ast, right=right_ast, op=n[1].ast_op)
            return ast_binop

        elif n[1].ast_type == ast.UnaryOp:
            operand = g.nodes_map[g.edges_end(n, label='operand')[0][0]]
            operand_ast = node_to_ast(operand, g, var_def)

            ast_unop = ast.UnaryOp(operand=operand_ast, op=n[1].ast_op)
            return ast_unop

        elif n[1].ast_type == ast.Call:

            args = [g.nodes_map[ii[0]] for ii in g.edges_end(n, label='args')]
            args_ast = []
            for a in args:
                a_ast = node_to_ast(a, g, var_def)
                args_ast.append(a_ast)

            ast_Call = ast.Call(args=args_ast, func=n[1].func, keywords={})

            return ast_Call

        elif n[1].ast_type == ast.IfExp:
            body = g.nodes_map[g.edges_end(n, label='body')[0][0]]
            body_ast = node_to_ast(body, g, var_def)

            orelse = g.nodes_map[g.edges_end(n, label='orelse')[0][0]]
            orelse_ast = node_to_ast(orelse, g, var_def)

            test = g.nodes_map[g.edges_end(n, label='test')[0][0]]
            test_ast = node_to_ast(test, g, var_def)

            ast_ifexp = ast.IfExp(body=body_ast, orelse=orelse_ast, test=test_ast)

            return ast_ifexp

        elif n[1].ast_type == ast.Compare:
            comp = [g.nodes_map[ii[0]] for ii in g.edges_end(n, label='comp')]
            comp_ast = []
            for a in comp:
                a_ast = node_to_ast(a, g, var_def)
                comp_ast.append(a_ast)

            left = g.nodes_map[g.edges_end(n, label='left')[0][0]]
            left_ast = node_to_ast(left, g, var_def)

            ast_Comp = ast.Compare(left=left_ast, comparators=comp_ast, ops=n[1].ops)

            return ast_Comp

        # TODO implement missing code ast objects
        raise TypeError(f'Cannot convert {n[1]},{n[1].ast_type}')
    except:
        print(n)
        raise


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

        return ast.Subscript(slice=ast.Index(value=ast.Num(n=ix)), value=ast.Name(id='l'))


def function_from_graph(g: Graph, name):
    lineno_count = 1

    top_nodes = g.topological_nodes()
    var_def_ = Vardef()
    var_def = var_def_.var_def



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

    decorators = [ast.Call(func=ast.Name(id='njit'), args=[ast.Str(s='void(float64[:])')], keywords={})]
    func = wrap_function(name, body, decorators=decorators, args=args)

    return func, var_def_.vars_inds_map

def switchboard_function(funcs_map):
    prev = None
    funcs = list(funcs_map.values())
    i=0
    for i, f in enumerate(funcs_map.values()):

        expr = ast.Expr(value=ast.Call(args=[ast.Name(id='locals')], func=ast.Name(id=f['func_name']), keywords={}))
        ifexp = ast.If(body=[expr], orelse=[],
                       test=ast.Compare(comparators=[ast.Num(n=i)], left=ast.Name(id='index'), ops=[ast.Eq()]))

        if not prev:

            out = ifexp

        else:
            prev.orelse.append(ifexp)

        prev = ifexp

    prev.orelse.append(ast.Raise(type=ast.Call(args=[ast.Str(s='Index out of bounds')], func=ast.Name(id='IndexError'), keywords={}), inst=None, tback=None))

    body=[out]



    args = dot_dict(args=[ast.Name(id='index'), ast.Name(id='locals')], vararg=None, defaults=[], kwarg=None)
    decorators = [ast.Call(func=ast.Name(id='njit'), args=[ast.Str(s='void(int64, float64[:])')], keywords={})]
    return wrap_function('switchboard', body, args,  decorators)

def generate(global_graph, vars_map):
    count=0
    mod_body = []
    known_graphs=set()
    print('creating ast')
    nodes = global_graph.get_nodes()

    funcs_map = {}
    derivative_map = {}

    def process_func(graph, func_name):
        if len(graph.edges) > 0:
            graph_hash = graph.hash()
            func_ast, var_map_func = function_from_graph(graph, func_name)

            mod_body.append(func_ast)

            nodes = [n[0] for n in graph.get_nodes()]
            graph_indcs = [nodes.index(v) for v in var_map_func]

            funcs_map[graph_hash] = {'index': count, 'func_name': func_name, 'var_map': var_map_func,
                                     'graph_indcs': graph_indcs}

    derivative_nodes = [n for n in nodes if hasattr(n[1].scope_var, 'type') and n[1].scope_var.type == VariableType.DERIVATIVE]
    l = len(derivative_nodes)
    print('l: ',l)
    for i, n in enumerate(derivative_nodes):

        #ancestor_graph = global_graph.get_ancestor_graph(n)
        #ancestor_graph.as_graphviz('problema')
        print(i / l * 100)

        #agn = [an for an in ancestor_graph.get_nodes() if hasattr(an[1].scope_var, 'type') and an[1].scope_var.type == VariableType.STATE]

        #dependants_graph = ancestor_graph.get_dependants_graph(agn)
        ancestor_graph, dependants_graph, deriv_dependencies = global_graph.get_ancestor_dependents_graph(n)
        #if i>-1:
        dependants_graph.as_graphviz(f'dep_anc{i}')
        ancestor_graph.as_graphviz(f'anc{i}')



        #print(dependants_graph.get_nodes())
        graph_hash = dependants_graph.hash()


        if not graph_hash in known_graphs:
            known_graphs.update([graph_hash])
            func_name = f'diff_{n[1].scope_var.tag}{count}'
            process_func(dependants_graph, func_name)

            count += 1
        #else:
        graph_indcs = funcs_map[graph_hash]['graph_indcs']
        nodes = [n[0] for n in dependants_graph.get_nodes()]

        #print( [nodes[i] for i in graph_indcs])
        
        derivative_map[n[0]] = {'node': n, 'func_hash': graph_hash, 'func': funcs_map[graph_hash]['func_name'], 'dependent_derivs': [d[0] for d in deriv_dependencies], 'variables_local_order': [nodes[i] for i in graph_indcs]}

                #signature


        #except:
            #dependants_graph.as_graphviz('problemd')
            #ancestor_graph.as_graphviz('problema')
         #   raise

    #Process derivatives into program

    constant_edges = []
    constant_nodes = []
    preprocess_edges = []
    preprocess_nodes = []

    gnodes = global_graph.nodes_map

    for e in global_graph.edges:
        if e[3] == 0:
            constant_edges.append(e)
            constant_nodes.append(gnodes[e[0]])
            constant_nodes.append(gnodes[e[1]])

        elif e[3] == 1:
            preprocess_edges.append(e)
            preprocess_nodes.append(gnodes[e[0]])
            preprocess_nodes.append(gnodes[e[1]])


    cg = Graph(constant_nodes, edges=constant_edges)
    cg.as_graphviz('constant')
    ppg = Graph(preprocess_nodes, edges=preprocess_edges)
    ppg.as_graphviz('preprocess')

    process_func(cg, 'constant_')
    process_func(ppg, 'preprocess_')



    mod_body.append(switchboard_function(funcs_map))


    de = []
    for d, data in derivative_map.items():
       de += [(dd, d, '') for dd in data['dependent_derivs']]


    nd = [d['node'] for d in derivative_map.values()]

    derivative_graph = Graph(nodes=nd, edges=de)

    funcs_indcs = list(funcs_map.keys())
    var_indcs = []
    program = []

    for d in derivative_graph.topological_nodes():
        start_ix = len(var_indcs)

        var_indcs += [vars_map.index(i[0]) for i in d[]]
        end_ix = len(var_indcs)
        print('fh',d['func_hash'])
        line = (funcs_indcs.index(d['func_hash']), start_ix, end_ix)
        program.append(line)

    print(program)

    derivative_graph.as_graphviz('deriv_dependecies')
    mod = wrap_module(mod_body)
    print('Generating Source')
    source = "from numba import njit, float64\nimport numpy as np\n" + astor.to_source(mod, indent_with=' ' * 4,
                                                                                       add_line_information=False,
                                                                                       source_generator_class=astor.SourceGenerator)

    with open('generated code.py', 'w') as f:
        f.write(source)


    print('Compiling')
    import timeit
    print('Compile time: ', timeit.timeit(
            lambda: exec(source, globals()),  number=1))


    #asd=sdfsdfsdf
