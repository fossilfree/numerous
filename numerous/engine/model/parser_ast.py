import inspect
import ast#, astor
from textwrap import dedent
from numerous.engine.model.graph import Graph
from numerous.engine.model.utils import NodeTypes, recurse_Attribute, Vardef, dot_dict, wrap_function, wrap_module
from numerous.engine.variables import VariableType

op_sym_map = {ast.Add: '+', ast.Sub: '-', ast.Div: '/', ast.Mult: '*', ast.Pow: '**', ast.USub: '*-1',
              ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}

def get_op_sym(op):
    return op_sym_map[type(op)]




def attr_ast(attr):
    attr_ = attr.split('.')
    # print(attr_)
    if len(attr_) >1:
        prev = None
        attr_str = attr_[-1]
        attr_=attr_[:-1]
        for a in attr_:
            if not prev:
                prev = ast.Name(id=a)
            else:
                prev = ast.Attribute(attr=a, value=prev)

        attr_ast = ast.Attribute(attr=attr_str, value=prev)
    else:
        # print('attr_[0]')
        # print(attr_[0])
        attr_ast = ast.Name(id=attr_[0])
    return attr_ast

# Add nodes and edges to a graph
tmp_count = [0]
def tmp(a):
    a+='_'+str(tmp_count[0])
    tmp_count[0]+=1
    return a

ass_count = [0]
def ass(a):
    a+='_'+str(ass_count[0])
    ass_count[0]+=1
    return a

class EquationNode:
    def __init__(self, ast, name, file, ln, label, ast_type, id=None, node_type:NodeTypes=NodeTypes.VAR, scope_var=None,**attrs):
        if not id:
            id = tmp(label)
        self.id = id
        self.label = label
        self.ast_type = ast_type
        self.scope_var=scope_var
        self.file = file
        self.func_name = name
        self.node_type=node_type


        if ast:
            self.lineno = ast.lineno + ln - 1
            self.col_offset= ast.col_offset
        else:
            self.lineno = None
            self.col_offset = None


        for k, v in attrs.items():
            setattr(self, k, v)

    def __str__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

class EquationEdge:
    def __init__(self, label, start: str = None, end: str = None):
        self.label = label
        self.start = start
        self.end = end

    def set_start(self, node_id: str):
        self.start = node_id

    def set_end(self, node_id: str):
        self.end = node_id

# Parse a function

def node_to_ast(n: EquationNode, g: Graph, var_def, read=True):
    try:
        if n[1].ast_type == ast.Attribute:
            return var_def(n[0], read)
        elif n[1].ast_type == ast.Name:
            return var_def(n[0], read)
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





def function_from_graph(g: Graph, name, decorators = [ast.Call(func=ast.Name(id='njit'), args=[ast.Str(s='void(float64[:])')], keywords={})]):
    lineno_count = 1

    top_nodes = g.topological_nodes()
    var_def_ = Vardef()
    var_def = var_def_.var_def



    body = []
    targets = []
    for n in top_nodes:
        lineno_count += 1

        if n[1].ast_type == ast.Assign or n[1].ast_type == ast.AugAssign:
            # n[1].id = n[0]
            value_node = g.nodes_map[g.edges_end(n, label='value')[0][0]]
            value_ast = node_to_ast(value_node, g, var_def)

            target_node = g.nodes_map[g.edges_start(n, label='target0')[0][1]]
            target_ast = node_to_ast(target_node, g, var_def)

            if value_ast and target_ast:
                if n[1].ast_type == ast.Assign or target_node not in targets:
                    targets.append(target_node)
                    ast_assign = ast.Assign(targets=[target_ast], value=value_ast)
                else:
                    ast_assign = ast.AugAssign(target=target_ast, value=value_ast, op=ast.Add())
                body.append(ast_assign)

    args = dot_dict(args=[ast.Name(id='l')], vararg=None, defaults=[], kwarg=None)


    func = wrap_function(name, body, decorators=decorators, args=args)

    return func, var_def_.vars_inds_map

def function_from_graph_generic(g: Graph, name, var_def_, decorators = [ast.Call(func=ast.Name(id='njit'), args=[ast.Str(s='void(float64[:])')], keywords={})]):
    lineno_count = 1

    top_nodes = g.topological_nodes()

    var_def = var_def_.var_def



    body = []
    targets = []
    for n in top_nodes:
        lineno_count += 1

        if n[1].ast_type == ast.Assign or n[1].ast_type == ast.AugAssign:
            # n[1].id = n[0]
            value_node = g.nodes_map[g.edges_end(n, label='value')[0][0]]
            value_ast = node_to_ast(value_node, g, var_def)

            target_node = g.nodes_map[g.edges_start(n, label='target0')[0][1]]
            target_ast = node_to_ast(target_node, g, var_def, read=False)

            if value_ast and target_ast:
                if n[1].ast_type == ast.Assign or target_node not in targets:
                    targets.append(target_node)
                    ast_assign = ast.Assign(targets=[target_ast], value=value_ast)
                else:
                    ast_assign = ast.AugAssign(target=target_ast, value=value_ast, op=ast.Add())
                body.append(ast_assign)

    if len(var_def_.get_targets())>1:
        return_ = ast.Return(value=ast.Tuple(elts=var_def_.get_targets()))
    else:
        return_ = ast.Return(value=var_def_.get_targets()[0])
    body.append(return_)
    args = dot_dict(args=var_def_.get_args(), vararg=None, defaults=[], kwarg=None)


    func = wrap_function(name, body, decorators=decorators, args=args)

    return func, var_def_.vars_inds_map

def parse_(ao, name, file, ln, g: Graph, tag_vars, prefix='_', parent: EquationEdge=None):
    # print(ao)
    en=None
    is_mapped = None

    if isinstance(ao, ast.Module):
        for b in ao.body:

            # Check if function def
            if isinstance(b, ast.FunctionDef):
                # Get name of function


                # Parse function
                for b_ in b.body:

                    parse_(b_, name, file, ln, g, tag_vars, prefix)

    elif isinstance(ao, ast.Assign):

        assert len(ao.targets) == 1, 'Can only parse assignments with one target'

        target = ao.targets[0]

        # Check if attribute
        if isinstance(ao.targets[0], ast.Attribute) or isinstance(ao.targets[0], ast.Name):

            att = recurse_Attribute(ao.targets[0])
            # print(att)
            target_id = att

        else:
            raise AttributeError('Unknown type of target: ', type(ao.targets[0]))

        target_edge = EquationEdge(label='target0')
        value_edge = EquationEdge(label='value')



        parse_(ao.value, name, file, ln, g, tag_vars,prefix, parent=value_edge.set_start)
        mapped = parse_(ao.targets[0], name, file, ln, g, tag_vars, prefix, parent=target_edge.set_end)

        en = EquationNode(ao, file, name, ln, label='+=' if mapped else '=', ast_type=ast.AugAssign if mapped else ast.Assign, node_type=NodeTypes.ASSIGN, ast_op=ast.Add() if mapped else None)
        g.add_node((en.id, en, None), ignore_exist=True)

        target_edge.start = en.id
        value_edge.end = en.id

        g.add_edge((value_edge.start, value_edge.end, value_edge, 0), ignore_missing_nodes=False)
        g.add_edge((target_edge.start, target_edge.end, target_edge, 0), ignore_missing_nodes=False)

    elif isinstance(ao, ast.Num):
        # Constant
        #source_var = Variable('c' + str(ao.value), Variable.CONSTANT, val=ao.value)
        source_id = 'c' + str(ao.value)
        en = EquationNode(ao, file, name, ln, label=source_id, ast_type=ast.Num, value = ao.value, node_type=NodeTypes.VAR)
        g.add_node((en.id, en, None), ignore_exist=True)

        # Check if simple name
    elif isinstance(ao, ast.Name) or isinstance(ao, ast.Attribute):
        local_id = recurse_Attribute(ao)

        source_id = local_id
        if source_id[:6]=='scope.':
            scope_var = tag_vars[source_id[6:]]


            #print('scope var: ',scope_var.id)
        else:
            scope_var=None

        if '-' in source_id:
            raise ValueError(f'Bad character -')

        if scope_var:
            if scope_var.type == VariableType.DERIVATIVE:
                node_type = NodeTypes.DERIV
            elif scope_var.type == VariableType.STATE:
                node_type = NodeTypes.STATE
            else:
                node_type = NodeTypes.VAR

            is_mapped = scope_var.sum_mapping_ids or scope_var.mapping_id

        else:
            node_type = NodeTypes.VAR

        en = EquationNode(ao, file, name, ln, id=source_id, local_id=local_id, ast_type=type(ao), label=source_id, node_type=node_type, scope_var=scope_var)
        g.add_node((en.id, en, None), ignore_exist=True)



    elif isinstance(ao, ast.UnaryOp):
        # Unary op
        op_sym = get_op_sym(ao.op)
        operand_edge = EquationEdge(label='operand')
        en = EquationNode(ao, file, name, ln, label = op_sym, ast_type=ast.UnaryOp, node_type=NodeTypes.OP, ast_op=ao.op)
        g.add_node((en.id, en, None), ignore_exist=True)
        operand_edge.end = en.id



        parse_(ao.operand, name, file, ln, g, tag_vars, prefix, parent=operand_edge.set_start)

        g.add_edge((operand_edge.start, operand_edge.end, operand_edge, 0), ignore_missing_nodes=False)

    elif isinstance(ao, ast.Call):


        op_name = recurse_Attribute(ao.func, sep='.')

        en = EquationNode(ao, file, name, ln, label=op_name, func=ao.func, ast_type=ast.Call, node_type=NodeTypes.OP)
        g.add_node((en.id, en, None), ignore_exist=True)


        for i, sa in enumerate(ao.args):
            edge_i = EquationEdge(end=en.id, label=f'args{i}')

            parse_(ao.args[i], name, file, ln, g, tag_vars, prefix=prefix, parent=edge_i.set_start)
            g.add_edge((edge_i.start, edge_i.end, edge_i, 0), ignore_missing_nodes=False)


    elif isinstance(ao, ast.BinOp):

        op_sym = get_op_sym(ao.op) # astor.get_op_symbol(ao.op)
        en = EquationNode(ao, file, name, ln, label=op_sym, left=None, right=None, ast_type=ast.BinOp, node_type=NodeTypes.OP, ast_op=ao.op)
        g.add_node((en.id, en, None), ignore_exist=True)
        for a in ['left', 'right']:

            operand_edge = EquationEdge(end=en.id, label=a)


            parse_(getattr(ao, a), name, file, ln, g, tag_vars, prefix, parent=operand_edge.set_start)

            g.add_edge((operand_edge.start, operand_edge.end, operand_edge, 0), ignore_missing_nodes=False)
            setattr(en, a, operand_edge)

    elif isinstance(ao, ast.Compare):
        ops_sym = [get_op_sym(o) for o in ao.ops]
        en = EquationNode(ao, file, name, ln, label=''.join(ops_sym), ast_type=ast.Compare, node_type=NodeTypes.OP, ops=ao.ops)
        g.add_node((en.id, en, None), ignore_exist=True)
        edge_l = EquationEdge(end=en.id, label=f'left')
        parse_(ao.left, name, file, ln, g, tag_vars, prefix=prefix, parent=edge_l.set_start)
        g.add_edge((edge_l.start, edge_l.end, edge_l, 0), ignore_missing_nodes=False)

        for i, sa in enumerate(ao.comparators):
            edge_i = EquationEdge(end=en.id, label=f'comp{i}')

            parse_(sa, name, file, ln, g, tag_vars, prefix=prefix, parent=edge_i.set_start)
            g.add_edge((edge_i.start, edge_i.end, edge_i, 0), ignore_missing_nodes=False)

    elif isinstance(ao, ast.IfExp):

       # astor.get_op_symbol(ao.op)
        en = EquationNode(ao, file, name, ln, label='if_exp', ast_type=ast.IfExp, node_type=NodeTypes.OP)
        g.add_node((en.id, en, None), ignore_exist=True)
        for a in ['body', 'orelse', 'test']:
            operand_edge = EquationEdge(end=en.id, label=a)


            parse_(getattr(ao, a), name, file, ln, g, tag_vars, prefix, parent=operand_edge.set_start)

            setattr(en, a, operand_edge)
            g.add_edge((operand_edge.start, operand_edge.end, operand_edge, 0), ignore_missing_nodes=False)

    else:
        raise TypeError('Cannot parse <' + str(type(ao)) + '>')




    if parent:

        parent(en.id)

    return is_mapped

def qualify(s, prefix):
    return prefix + '_' + s.replace('scope.', '')

def qualify_equation(prefix, g, tag_vars):
    #for k, v in tag_vars.items():
    #    if '-' in v.id:
     #       print(v.id)
     #       raise ValueError('arg!')

    def q(s):
        return qualify(s, prefix)

    g_qual = Graph()
    g_qual.set_node_map({q(n[0]): (q(n[0]), n[1], (tag_vars[n[1].scope_var.tag].id if n[1].scope_var else None)) for nid, n in g.nodes_map.items()})

    #nodes_parents = {}
    #for n in g_qual.get_nodes():
    #    nodes_parents[n[0]] = [(e[0],) for e in g_qual.edges if e[1]==n[0]]

    for e in g.edges:
        g_qual.add_edge((q(e[2].start),q(e[2].end), EquationEdge(start=q(e[2].start), end=q(e[2].end), label=e[2].label), 0))

    return g_qual


def parse_eq(scope_id, item, global_graph, equation_graph, nodes_dep, tag_vars, parsed_eq, scoped_equations):
    #print(item)

    for eq in item[0]:

        #dont now how Kosher this is: https://stackoverflow.com/questions/20059011/check-if-two-python-functions-are-equal
        eq_key = eq.__qualname__
        #print(eq_key)



        if not eq_key in parsed_eq:
            #print('parsing')
            dsource = inspect.getsource(eq)

            tries=0
            while tries<5:
                try:
                    dsource = dedent(dsource)
                    ast_tree = ast.parse(dsource)
                    break
                except IndentationError:
                    tries+=1
                    if tries > 5-1:
                        raise




            g = Graph()

            parse_(ast_tree, eq.__qualname__, eq.file, eq.lineno, g, tag_vars)

            parsed_eq[eq_key] = (eq, dsource, g)



        else:
            pass
            #print('skip parsing')

        g = parsed_eq[eq_key][2]
        #print('qualify with ',scope_id)
        g_qualified = qualify_equation(scope_id, g, tag_vars)
        g_qualified.make_lower_graph(top_sort=False)



        nodes = g_qualified.get_nodes()

        #make equation graph
        eq_name = ('EQ_'+scope_id + '_' + eq_key).replace('.','_')

        scoped_equations[eq_name] = eq_key

        equation_graph.add_node((eq_name, EquationNode(id=eq_name, node_type=NodeTypes.EQUATION, ast=None, name=eq_name, file=eq_name, ln=0, label=eq_name, ast_type=ast.Call, func=ast.Name(id=eq_key.replace('.','_')), op_type=ast.Call), eq_name), ignore_exist=True)
        #edges = []
        #eg_nodes = [eq_name]
        for n in nodes:
            if n[1].node_type == NodeTypes.VAR or n[1].node_type == NodeTypes.DERIV or n[1].node_type == NodeTypes.STATE:

                if not n[0] in nodes_dep:
                    nodes_dep[n[0]] = []
                if not eq_name in nodes_dep[n[0]]:
                    nodes_dep[n[0]].append(eq_name)

                #in_deg = in_degree[n[0]]
                #eg_nodes.append(n[0])
                equation_graph.add_node(n, ignore_exist=True)
                #if in_deg > 0:
                #    equation_graph.add_edge((eq_name, n[0], 'a', 0), ignore_missing_nodes=False)
                #else:
                #    equation_graph.add_edge((n[0], eq_name, 'a', 0), ignore_missing_nodes=False)
                targeted = False
                read = False
                for e in g_qualified.edges:
                    print(e)
                    if e[1] == n[0]:
                        equation_graph.add_edge((eq_name, n[0], EquationEdge(start=eq_name, end=n[0], label='target'), 0), ignore_missing_nodes=False)
                        targeted = True
                        break
                if not targeted and not read:
                    for e in g_qualified.edges:
                        if e[0] == n[0]:
                            read=True

                            equation_graph.add_edge((n[0], eq_name, EquationEdge(start=eq_name, end=n[0], label='args'), 0), ignore_missing_nodes=False)
                            break



        #eg = Graph(nodes=eg_nodes, edges=edges)
        #equation_graph.update(eg)




        global_graph.update(g_qualified)
        a = 1



def process_mappings(mappings,gg:Graph, equation_graph:Graph, nodes_dep, scope_vars, scope_map):
    mg = Graph()
    for m in mappings:
        target_var = scope_vars[m[0]]
        #prefix = scope_map[target_var.parent_scope_id]
        prefix = scope_map[target_var.parent_scope_id]


        target_var_id = qualify(target_var.tag, prefix)

        if '-' in target_var_id:
            raise ValueError('argh')

        if target_var.type == VariableType.DERIVATIVE:
            node_type = NodeTypes.DERIV
        elif target_var.type == VariableType.STATE:
            node_type = NodeTypes.STATE
        else:
            node_type = NodeTypes.VAR

        assign = EquationNode(None, file='mapping', name=m, ln=0, label='=', ast_type=ast.AugAssign, node_type=NodeTypes.ASSIGN, targets=[], value=None, ast_op=ast.Add())
        gg.add_node((assign.id, assign, None))
        equation_graph.add_node((assign.id, assign, None))


        target_node = EquationNode(None, file='mapping', name=m, ln=0, id=target_var_id, label=target_var.tag, ast_type=ast.Attribute, node_type=node_type, scope_var=target_var)
        gg.add_node((target_node.id, target_node, target_var.id), ignore_exist=True)
        equation_graph.add_node((target_node.id, target_node, target_var.id), ignore_exist=True)

        if not target_var_id in nodes_dep:
            nodes_dep[target_var_id] = []
        if not assign.id in nodes_dep[target_var_id]:
            nodes_dep[target_var_id].append(assign.id)

        gg.add_edge((assign.id, target_node.id, EquationEdge(start=assign.id, end=target_node.id, label='target0'), 0), ignore_missing_nodes=False)
        equation_graph.add_edge((assign.id, target_node.id, EquationEdge(start=assign.id, end=target_node.id, label='target0'), 0),
                    ignore_missing_nodes=False)

        add = ast.Add()
        prev = None

        mg.add_node((target_var.parent_scope_id, None), ignore_exist=True)

        for i in m[1]:
            ivar_var = scope_vars[i]
            #prefix = f's{scope_map.index(ivar_var.parent_scope_id)}'
            prefix = scope_map[ivar_var.parent_scope_id]
            mg.add_node((ivar_var.parent_scope_id, None), ignore_exist=True)
            mg.add_edge((ivar_var.parent_scope_id, target_var.parent_scope_id,'mapping'), 0)

            ivar_id = qualify(ivar_var.tag, prefix)

            if not ivar_id in nodes_dep:
                nodes_dep[ivar_id] = []
            if not assign.id in nodes_dep[ivar_id]:
                nodes_dep[ivar_id].append(assign.id)



            if '-' in ivar_id:
                raise ValueError('argh')

            ivar = EquationNode(None, file='mapping', name=m, ln=0, id=ivar_id, label=ivar_var.tag, ast_type=ast.Attribute, node_type=NodeTypes.VAR)
            gg.add_node((ivar.id, ivar, ivar_var.id, ), ignore_exist=True)
            equation_graph.add_node((ivar.id, ivar, ivar_var.id,), ignore_exist=True)
            if prev:
                binop = EquationNode(None, file='mapping', name=m, ln=0, label=get_op_sym(add), ast_type=ast.BinOp, node_type=NodeTypes.OP, ast_op=add)
                gg.add_node((binop.id, binop, None,))
                gg.add_edge((prev.id, binop.id, EquationEdge(start=prev.id, end=binop.id,label='left'), 0), ignore_missing_nodes=False)
                gg.add_edge((ivar.id, binop.id, EquationEdge(start=ivar.id, end=binop.id,label='right'), 0), ignore_missing_nodes=False)

                equation_graph.add_node((binop.id, binop, None,))
                equation_graph.add_edge((prev.id, binop.id, EquationEdge(start=prev.id, end=binop.id, label='left'), 0),
                            ignore_missing_nodes=False)
                equation_graph.add_edge((ivar.id, binop.id, EquationEdge(start=ivar.id, end=binop.id, label='right'), 0),
                            ignore_missing_nodes=False)

                prev = binop
            else:
                prev = ivar

            #equation_graph.add_edge((ivar.id, target_node.id, 'mapping', 0), ignore_missing_nodes=False)

        gg.add_edge((prev.id, assign.id, EquationEdge(start=prev.id, end=assign.id, label='value'), 0), ignore_missing_nodes=False)
        equation_graph.add_edge((prev.id, assign.id, EquationEdge(start=prev.id, end=assign.id, label='value'), 0),
                    ignore_missing_nodes=False)

        #ast.Assign(targets=ast.Attribute(attr_ast(m[0])), value = None)

    #replace all mappings

    #Loop over all nodes
    #for n in equation_graph.get_nodes()

    equation_graph_simplified = equation_graph.clone()
    print('nodes dep: ')
    for nd, dep in nodes_dep.items():
        print(nd,': ', dep)
    for n, dep in nodes_dep.items():
        if len(dep) <= 1:
            # remove node and its edges
            print('remove: ', n)
            equation_graph_simplified.remove_node(n)
            edges = equation_graph_simplified.edges.copy()
            for e in edges:
                if e[0] == n or e[1] == n:
                    equation_graph_simplified.remove_edge(e)
    return equation_graph_simplified
    #mg.as_graphviz('mappings')




