import inspect
import ast#, astor
from textwrap import dedent
from numerous.engine.model.graph import Graph
from numerous.engine.model.utils import NodeTypes, recurse_Attribute, Vardef, dot_dict, wrap_function, wrap_module, VarTypes
from numerous.engine.variables import VariableType
from numerous.engine.scope import ScopeVariable

import logging

op_sym_map = {ast.Add: '+', ast.Sub: '-', ast.Div: '/', ast.Mult: '*', ast.Pow: '**', ast.USub: '*-1',
              ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}

def get_op_sym(op):
    return op_sym_map[type(op)]

def attr_ast(attr):
    attr_ = attr.split('.')
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

# Parse a function

def node_to_ast(n: int, g: Graph, var_def, read=True):
    nk = g.key_map[n]
    try:
        if (na:=g.get(n,'ast_type')) == ast.Attribute:
            return var_def(nk, read)

        elif na == ast.Name:
            return var_def(nk, read)

        elif na == ast.Num:
            return ast.Call(args=[ast.Num(value=g.get(n, 'value'))], func=ast.Name(id='float32'), keywords={})

        elif na == ast.BinOp:

            #left_node = g.nodes_map[g.edges_end(n, label='left',max=1)[0][0]]
            left_node = g.get_edges_for_node_filter(end_node=n, attr='e_type', val='left')[1][0][0]

            left_ast = node_to_ast(left_node, g, var_def)

            #right_node = g.nodes_map[g.edges_end(n, label='right',max=1)[0][0]]
            right_node = g.get_edges_for_node_filter(end_node=n, attr='e_type', val='right')[1][0][0]

            right_ast = node_to_ast(right_node, g, var_def)

            ast_binop = ast.BinOp(left=left_ast, right=right_ast, op=g.get(n, 'ast_op'))
            return ast_binop

        elif na == ast.UnaryOp:
            #operand = g.nodes_map[g.edges_end(n, label='operand',max=1)[0][0]]
            operand = g.get_edges_for_node_filter(end_node=n, attr='e_type', val='operand')[1][0][0]

            operand_ast = node_to_ast(operand, g, var_def)

            ast_unop = ast.UnaryOp(operand=operand_ast, op=g.get(n, 'ast_op'))
            return ast_unop

        elif na == ast.Call:

            args = [ii[0] for ii in g.get_edges_for_node_filter(end_node=n, attr='e_type', val='args')[1]]
            args_ast = []
            for a in args:
                a_ast = node_to_ast(a, g, var_def)
                args_ast.append(a_ast)

            ast_Call = ast.Call(args=args_ast, func=g.get(n, 'func'), keywords={})

            return ast_Call

        elif na == ast.IfExp:

            body = g.get_edges_for_node_filter(end_node=n, attr='e_type', val='body')[1][0][0]
            body_ast = node_to_ast(body, g, var_def)

            orelse = g.get_edges_for_node_filter(end_node=n, attr='e_type', val='orelse')[1][0][0]
            orelse_ast = node_to_ast(orelse, g, var_def)

            test = g.get_edges_for_node_filter(end_node=n, attr='e_type', val='test')[1][0][0]
            test_ast = node_to_ast(test, g, var_def)

            ast_ifexp = ast.IfExp(body=body_ast, orelse=orelse_ast, test=test_ast)

            return ast_ifexp

        elif na == ast.Compare:
            comp = [ii[0] for ii in g.get_edges_for_node_filter(end_node=n, attr='e_type', val='comp')[1]]
            comp_ast = []
            for a in comp:
                a_ast = node_to_ast(a, g, var_def)
                comp_ast.append(a_ast)

            #left = g.nodes_map[g.edges_end(n, label='left',max=1)[0][0]]
            left = g.get_edges_for_node_filter(end_node=n, attr='e_type', val='left')[1][0][0]

            left_ast = node_to_ast(left, g, var_def)

            ast_Comp = ast.Compare(left=left_ast, comparators=comp_ast, ops=g.get(n,'ops'))

            return ast_Comp

        # TODO implement missing code ast objects
        raise TypeError(f'Cannot convert {n},{na}')
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

        if (na:=g.get(n,'ast_type')) == ast.Assign or na == ast.AugAssign:
            # n[1].id = n[0]
            value_node = g.nodes_map[g.edges_end(n, label='value',max=1)[0][0]]
            value_ast = node_to_ast(value_node, g, var_def)

            target_node = g.nodes_map[g.edges_start(n, label='target0',max=1)[0][1]]
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

        if (at:=g.get(n, 'ast_type')) == ast.Assign or at == ast.AugAssign:
            # n[1].id = n[0]
            #value_node = g.nodes_map[g.edges_end(n, label='value',max=1)[0][0]]

            value_node = g.get_edges_for_node_filter(end_node=n, attr='e_type', val='value')[1][0][0]

            value_ast = node_to_ast(value_node, g, var_def)

            #target_node = g.nodes_map[g.edges_start(n, label='target0',max=1)[0][1]]
            target_node = g.get_edges_for_node_filter(start_node=n, attr='e_type', val='target')[1][0][1]
            target_ast = node_to_ast(target_node, g, var_def, read=False)

            if value_ast and target_ast:
                if (na:=g.get(n,'ast_type')) == ast.Assign or target_node not in targets:
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

def function_from_graph_generic_llvm(g: Graph, name, var_def_):
    fname = name + '_llvm'

    lineno_count = 1

    top_nodes = g.topological_nodes()

    var_def = var_def_.var_def

    body = []
    targets = []
    for n in top_nodes:
        lineno_count += 1

        if (na:=g.get(n,'ast_type')) == ast.Assign or na == ast.AugAssign:
            # n[1].id = n[0]
            value_node = g.get_edges_for_node_filter(end_node=n, attr='e_type', val='value')[1][0][0]
            value_ast = node_to_ast(value_node, g, var_def)

            target_node = g.get_edges_for_node_filter(start_node=n, attr='e_type', val='target')[1][0][1]
            target_ast = node_to_ast(target_node, g, var_def, read=False)

            if value_ast and target_ast:
                if na == ast.Assign or target_node not in targets:
                    targets.append(target_node)
                    ast_assign = ast.Assign(targets=[target_ast], value=value_ast)
                else:
                    ast_assign = ast.AugAssign(target=target_ast, value=value_ast, op=ast.Add())
                body.append(ast_assign)

    len_targs = len(var_def_.get_targets())

    args = dot_dict(args=var_def_.get_args() + var_def_.get_targets(), vararg=None, defaults=[], kwarg=None)
    signature = f'void({", ".join(["float32" for a in var_def_.get_args()])}, {", ".join(["CPointer(float32)" for a in var_def_.get_targets()])})'
    decorators = []

    func = wrap_function(fname, body, decorators=decorators, args=args)



    return func, var_def_.vars_inds_map, signature, fname, var_def_.args, var_def_.targets

def parse_(ao, name, file, ln, g: Graph, tag_vars, prefix='_'):
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
            target_id = att

        else:
            raise AttributeError('Unknown type of target: ', type(ao.targets[0]))





        m, start = parse_(ao.value, name, file, ln, g, tag_vars,prefix)
        mapped, end = parse_(ao.targets[0], name, file, ln, g, tag_vars, prefix)

        #en = EquationNode(ao, file, name, ln, label='+=' if mapped else '=', ast_type=ast.AugAssign if mapped else ast.Assign, node_type=NodeTypes.ASSIGN, ast_op=ast.Add() if mapped else None)
        en = g.add_node(ao=ao, file=file, name=name, ln=ln, label='+=' if mapped else '=', ast_type=ast.AugAssign if mapped else ast.Assign, node_type=NodeTypes.ASSIGN, ast_op=ast.Add() if mapped else None)
        target_edge = g.add_edge(start=en, end=end, e_type='target')
        value_edge = g.add_edge(start=start, end=en, e_type='value')
        #g.set_edge(target_edge, start=en)
        #g.set_edge(value_edge, end=en)

        #g.add_edge((value_edge.start, value_edge.end, value_edge, 0), ignore_missing_nodes=False)
        #g.add_edge((target_edge.start, target_edge.end, target_edge, 0), ignore_missing_nodes=False)

    elif isinstance(ao, ast.Num):
        # Constant
        #source_var = Variable('c' + str(ao.value), Variable.CONSTANT, val=ao.value)
        source_id = 'c' + str(ao.value)
        en = g.add_node(ao=ao, file=file, name=name, ln=ln, label=source_id, ast_type=ast.Num, value = ao.value, node_type=NodeTypes.VAR)
        #g.add_node(en.id, equation_node=en, ignore_existing=True)

        # Check if simple name
    elif isinstance(ao, ast.Name) or isinstance(ao, ast.Attribute):
        local_id = recurse_Attribute(ao)

        source_id = local_id
        if source_id[:6]=='scope.':
            scope_var = tag_vars[source_id[6:]]

        else:
            scope_var=None

        if '-' in source_id:
            raise ValueError(f'Bad character -')

        node_type = NodeTypes.VAR

        if scope_var:



            var_type = VariableType.DERIVATIVE
            is_mapped = scope_var.sum_mapping_ids or scope_var.mapping_id

        else:
            var_type = VariableType.PARAMETER


        en = g.add_node(key=source_id, ao=ao, file=file, name=name, ln=ln, id=source_id, local_id=local_id, ast_type=type(ao), node_type=node_type, scope_var=scope_var, ignore_existing=True)


    elif isinstance(ao, ast.UnaryOp):
        # Unary op
        op_sym = get_op_sym(ao.op)

        en = g.add_node(label=op_sym, ast_type=ast.UnaryOp, node_type=NodeTypes.OP, ast_op=ao.op, ignore_existing=True)

        m, start = parse_(ao.operand, name, file, ln, g, tag_vars, prefix)
        operand_edge = g.add_edge(start=start, e_type='operand', end=en)

    elif isinstance(ao, ast.Call):

        op_name = recurse_Attribute(ao.func, sep='.')

        en = g.add_node(ao=ao, file=file, name=name, ln=ln, label=op_name, func=ao.func, ast_type=ast.Call, node_type=NodeTypes.OP)


        for i, sa in enumerate(ao.args):

            m, start = parse_(ao.args[i], name, file, ln, g, tag_vars, prefix=prefix)
            g.add_edge(start=start, end=en, e_type='args')


    elif isinstance(ao, ast.BinOp):

        op_sym = get_op_sym(ao.op) # astor.get_op_symbol(ao.op)
        en = g.add_node(ao=ao, file=file, name=name, ln=ln, label=op_sym, left=None, right=None, ast_type=ast.BinOp, node_type=NodeTypes.OP, ast_op=ao.op)

        for a in ['left', 'right']:

            m, start = parse_(getattr(ao, a), name, file, ln, g, tag_vars, prefix)
            operand_edge = g.add_edge(start=start, end=en, e_type=a)

    elif isinstance(ao, ast.Compare):
        ops_sym = [get_op_sym(o) for o in ao.ops]

        en = g.add_node(ao=ao, file=file, name=name, ln=ln, label=''.join(ops_sym), ast_type=ast.Compare, node_type=NodeTypes.OP, ops=ao.ops)


        m, start = parse_(ao.left, name, file, ln, g, tag_vars, prefix=prefix)

        edge_l = g.add_edge(start=start, end=en, label=f'left', e_type='left')

        for i, sa in enumerate(ao.comparators):

            m, start = parse_(sa, name, file, ln, g, tag_vars, prefix=prefix)
            edge_i = g.add_edge(start=start, end=en, label=f'comp{i}', e_type='comp')

    elif isinstance(ao, ast.IfExp):

        en = g.add_node(ao=ao, file=file, name=name, ln=ln, label='if_exp', ast_type=ast.IfExp, node_type=NodeTypes.OP)
        for a in ['body', 'orelse', 'test']:

            m, start = parse_(getattr(ao, a), name, file, ln, g, tag_vars, prefix)

            operand_edge = g.add_edge(start=start, end=en, e_type=a)

    else:
        raise TypeError('Cannot parse <' + str(type(ao)) + '>')

    return is_mapped, en

def qualify(s, prefix):
    return prefix + '_' + s.replace('scope.', '')

def qualify_equation(prefix, g, tag_vars):

    def q(s):
        return qualify(s, prefix)

    g_qual = g.clone()

    #update keys
    g_qual.node_map = {q(k): v for k,v in g_qual.node_map.items()}
    g_qual.key_map = {k: q(v) for k, v in g_qual.key_map.items()}
    g_qual.nodes_attr['scope_var'][:g_qual.node_counter] = [tag_vars[sv.tag] if isinstance(sv:=g.get(n, 'scope_var'), ScopeVariable) else sv for n in g.node_map.values()]

    return g_qual


def parse_eq(scope_id, item, global_graph, equation_graph: Graph, nodes_dep, tag_vars, parsed_eq, scoped_equations):

    for eq in item[0]:

        #dont now how Kosher this is: https://stackoverflow.com/questions/20059011/check-if-two-python-functions-are-equal
        eq_key = eq.__qualname__

        if not eq_key in parsed_eq:
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

            g.as_graphviz(eq_key)

        g = parsed_eq[eq_key][2]
        g_qualified = qualify_equation(scope_id, g, tag_vars)

        #make equation graph
        eq_name = ('EQ_'+scope_id + '_' + eq_key).replace('.','_')

        scoped_equations[eq_name] = eq_key

        eq_n = equation_graph.add_node(key=eq_name, node_type=NodeTypes.EQUATION, ast=None, name=eq_name, file=eq_name, ln=0, label=eq_name, ast_type=ast.Call, func=ast.Name(id=eq_key.replace('.','_')))

        for n in range(g_qualified.node_counter):

            if g_qualified.get(n, attr='node_type')==NodeTypes.VAR and g_qualified.get(n, attr='scope_var'):

                n_key = g_qualified.key_map[n]

                if not n_key in nodes_dep:
                    nodes_dep[n_key] = []
                if not eq_name in nodes_dep[n_key]:
                    nodes_dep[n_key].append(eq_name)

                sv = g_qualified.get(n, 'scope_var')

                neq = equation_graph.add_node(key=n_key, node_type=NodeTypes.VAR, scope_var=sv, ignore_existing=True)

                targeted = False
                read = False

                end_edges = g_qualified.get_edges_for_node(end_node=n)

                try:
                    next(end_edges)
                    equation_graph.add_edge(eq_n, neq, e_type='target', arg_local= sv.tag if sv else 'local')
                    targeted = True
                except StopIteration:
                    pass

                if not targeted and not read:
                    start_edges = g_qualified.get_edges_for_node(start_node=n)
                    try:
                        next(start_edges)
                        read=True
                        equation_graph.add_edge(neq, eq_n, e_type='arg', arg_local= sv.tag if (sv:= g_qualified.get(n, 'scope_var')) else 'local')
                    except StopIteration:
                        pass

        global_graph.update(g_qualified)
        a = 1

def process_mappings(mappings,gg:Graph, equation_graph:Graph, nodes_dep, scope_vars, scope_map):
    mg = Graph()
    logging.info('process mappings')
    for m in mappings:
        target_var = scope_vars[m[0]]
        prefix = scope_map[target_var.parent_scope_id]
        target_var_id = qualify(target_var.tag, prefix)

        if '-' in target_var_id:
            raise ValueError('argh')

        node_type = NodeTypes.VAR

        atmp=tmp('=')
        ag = gg.add_node(key=atmp, file='mapping', name=m, ln=0, label='=', ast_type=ast.AugAssign, node_type=NodeTypes.ASSIGN, targets=[], value=None, ast_op=ast.Add())
        ae = equation_graph.add_node(key=atmp,file='mapping', name=m, ln=0, label='=', ast_type=ast.AugAssign, node_type=NodeTypes.ASSIGN,
                    targets=[], value=None, ast_op=ast.Add())

        tg = gg.add_node(key=target_var_id , file='mapping', name=m, ln=0, id=target_var_id, label=target_var.tag, ast_type=ast.Attribute, node_type=node_type, scope_var=target_var, ignore_existing=True)
        t = equation_graph.add_node(key=target_var_id, file='mapping', name=m, ln=0, id=target_var_id, label=target_var.tag, ast_type=ast.Attribute, node_type=node_type, scope_var=target_var, ignore_existing=True)

        ak = equation_graph.key_map[ae]
        if not target_var_id in nodes_dep:
            nodes_dep[target_var_id] = []
        if not ak in nodes_dep[target_var_id]:
            nodes_dep[target_var_id].append(ak)

        gg.add_edge(start=ag, end=tg, e_type='target')
        equation_graph.add_edge(start=ae, end=t, e_type='target')

        add = ast.Add()
        prev = None

        tn = mg.add_node(key=target_var.parent_scope_id, ignore_existing=True, label='f(x)')

        for i in m[1]:
            ivar_var = scope_vars[i]
            prefix = scope_map[ivar_var.parent_scope_id]
            ivarn = mg.add_node(key=ivar_var.parent_scope_id, ignore_existing=True, label=ivar_var.parent_scope_id)
            mg.add_edge(start=ivarn, end=tn, e_type='mapping')

            ivar_id = qualify(ivar_var.tag, prefix)

            if not ivar_id in nodes_dep:
                nodes_dep[ivar_id] = []
            if not ae in nodes_dep[ivar_id]:
                nodes_dep[ivar_id].append(ae)

            if '-' in ivar_id:
                raise ValueError('argh')

            scope_var = scope_vars[ivar_var.id]

            ivar_node_g = gg.add_node(key=ivar_id, file='mapping', name=m, ln=0, id=ivar_id, label=ivar_var.tag, ast_type=ast.Attribute, node_type=NodeTypes.VAR, scope_var=scope_var, ignore_existing=True)
            ivar_node_e = equation_graph.add_node(key=ivar_id, file='mapping', name=m, ln=0, id=ivar_id, label=ivar_var.tag,
                                      ast_type=ast.Attribute, node_type=NodeTypes.VAR, scope_var=scope_var, ignore_existing=True)

            if prev:

                binop_g = gg.add_node(file='mapping', name=m, ln=0, label=get_op_sym(add), ast_type=ast.BinOp,
                                     node_type=NodeTypes.OP, ast_op=add)

                gg.add_edge(prev_g, binop_g, e_type='left')
                gg.add_edge(ivar_node_g, binop_g, e_type='right')

                binop_e = gg.add_node(file='mapping', name=m, ln=0, label=get_op_sym(add), ast_type=ast.BinOp,
                                      node_type=NodeTypes.OP, ast_op=add)

                equation_graph.add_edge(prev_e, binop_e, e_type='left')
                equation_graph.add_edge(ivar_node_e, binop_e, e_type='right')

                prev_g = binop_g
                prev_e = binop_e
            else:
                prev_g = ivar_node_g
                prev_e = ivar_node_e

        gg.add_edge(prev_g, ag,e_type='value')
        equation_graph.add_edge(prev_e, ae, e_type='value')

    gg.as_graphviz('global')
    equation_graph.as_graphviz('eq', force=True)
    if True:
        logging.info('making substituation graph')

        #replace all mappings
        substitutions = {}

        #Loop over all nodes
        eq_val_nodes = equation_graph.get_where_attr('node_type', NodeTypes.VAR)
        s_nodes = []
        s_edges = []

        substitution_graph = Graph()
        for n in eq_val_nodes:
                #targeting_edges = equation_graph.edges_end(n, 'target')
                ix, targeting_edges = equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val='target')

                #If only targeted once we can remap this!
                if len(targeting_edges) == 1:
                    op= targeting_edges[0][0]
                    if equation_graph.get(targeting_edges[0][0], 'node_type') == NodeTypes.ASSIGN:
                        ix, values = equation_graph.get_edges_for_node_filter(end_node=op, attr='e_type', val='value')#edges_end(op, 'value')
                        #add a map
                        if len(values) == 1:
                            val=values[0][0]

                            if not equation_graph.get(val,'ast_type') == ast.BinOp:
                                if not n in substitutions:
                                    val_k = equation_graph.key_map[val]
                                    substitutions[n] = val_k
                                    nm = substitution_graph.add_node(key=equation_graph.key_map[n], label=equation_graph.get(n, 'label'), node_type=0, ignore_existing=True)
                                    #mapped_to = equation_graph.nodes_map[values[0][0]]
                                    nv = substitution_graph.add_node(key=val_k, label=val_k, node_type=0, ignore_existing=True)

                                    substitution_graph.add_edge(nv, nm, e_type='value')
                                else:
                                    raise ValueError(equation_graph.key_map[n]+' already substituted???')

                                equation_graph.remove_node(op)
                                equation_graph.remove_node(n)

                        elif len(values) > 1:
                            raise ValueError('arg')

        substitution_graph.as_graphviz('subgraph', force=True)
        logging.info('cleaning')

        logging.info('define master vars')
        master_variables = substitution_graph.zero_in_degree()

        logging.info('process master vars')

        for mv in master_variables:
            dep_g = substitution_graph.get_dependants_graph(mv)

            mk = substitution_graph.key_map[mv]
            dep_g.node_map.pop(mk)

            equation_graph.replace_nodes_by_key(mk, dep_g.node_map.keys())

    logging.info('clone eq graph')
    equation_graph_simplified = equation_graph.clone()
    equation_graph_simplified.clean()
    equation_graph_simplified.as_graphviz('eqs', force=True)

    logging.info('remove dependencies')

    logging.info('cleaning')

    logging.info('done cleaning')

    return equation_graph_simplified




