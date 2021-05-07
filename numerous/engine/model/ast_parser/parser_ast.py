import ast
import logging
from copy import deepcopy
from textwrap import dedent

from graph_representation import Edge
from numerous.engine.model.graph_representation import MappingsGraph, Graph, EdgeType,Node
from numerous.engine.model.graph_representation.utils import str_to_edgetype
from numerous.engine.model.utils import NodeTypes, recurse_Attribute, wrap_function
from numerous.engine.scope import ScopeVariable
from numerous.engine.variables import VariableType

op_sym_map = {ast.Add: '+', ast.Sub: '-', ast.Div: '/', ast.Mult: '*', ast.Pow: '**', ast.USub: '*-1',
              ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}


def get_op_sym(op):
    return op_sym_map[type(op)]


def attr_ast(attr):
    attr_ = attr.split('.')
    if len(attr_) > 1:
        prev = None
        attr_str = attr_[-1]
        attr_ = attr_[:-1]
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
    a += '_' + str(tmp_count[0])
    tmp_count[0] += 1
    return a


ass_count = [0]


def ass(a):
    a += '_' + str(ass_count[0])
    ass_count[0] += 1
    return a


# Parse a function
def node_to_ast(n: int, g: MappingsGraph, var_def, ctxread=False, read=True):
    nk = g.key_map[n]
    try:
        if (na := g.get(n, 'ast_type')) == ast.Attribute:

            return var_def(nk, ctxread, read)

        elif na == ast.Name:
            return var_def(nk, ctxread, read)

        elif na == ast.Num:
            return ast.Call(args=[ast.Num(value=g.get(n, 'value'),
                                          lineno=0, col_offset=0)], func=ast.Name(id='float64', lineno=0, col_offset=0,
                                                                                  ctx=ast.Load()),
                            keywords=[], lineno=0, col_offset=0)

        elif na == ast.BinOp:

            left_node = g.get_edges_for_node_filter(end_node=n, attr='e_type', val=[EdgeType.LEFT])[1][0][0]

            left_ast = node_to_ast(left_node, g, var_def, ctxread=ctxread)

            right_node = g.get_edges_for_node_filter(end_node=n, attr='e_type', val=[EdgeType.RIGHT])[1][0][0]

            right_ast = node_to_ast(right_node, g, var_def, ctxread=ctxread)

            ast_binop = ast.BinOp(left=left_ast, right=right_ast, op=g.get(n, 'ast_op'), lineno=0, col_offset=0)
            return ast_binop

        elif na == ast.UnaryOp:
            operand = g.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.OPERAND)[1][0][0]

            operand_ast = node_to_ast(operand, g, var_def, ctxread=ctxread)

            ast_unop = ast.UnaryOp(operand=operand_ast, op=g.get(n, 'ast_op'), lineno=0, col_offset=0)
            return ast_unop

        elif na == ast.Call:

            args = [ii[0] for ii in g.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.ARGUMENT)[1]]
            args_ast = []
            for a in args:
                a_ast = node_to_ast(a, g, var_def, ctxread=ctxread)
                args_ast.append(a_ast)

            ast_Call = ast.Call(args=args_ast, func=g.get(n, 'func'), keywords=[], lineno=0, col_offset=0)

            return ast_Call

        elif na == ast.IfExp:

            body = g.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.BODY)[1][0][0]
            body_ast = node_to_ast(body, g, var_def, ctxread=ctxread)

            orelse = g.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.ORELSE)[1][0][0]
            orelse_ast = node_to_ast(orelse, g, var_def, ctxread=ctxread)

            test = g.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.TEST)[1][0][0]
            test_ast = node_to_ast(test, g, var_def, ctxread=ctxread)

            ast_ifexp = ast.IfExp(body=body_ast, orelse=orelse_ast, test=test_ast, lineno=0, col_offset=0)

            return ast_ifexp

        elif na == ast.Compare:
            comp = [ii[0] for ii in g.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.COMP)[1]]
            comp_ast = []
            for a in comp:
                a_ast = node_to_ast(a, g, var_def, ctxread=ctxread)
                comp_ast.append(a_ast)

            left = g.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.LEFT)[1][0][0]

            left_ast = node_to_ast(left, g, var_def, ctxread=ctxread)

            ast_Comp = ast.Compare(left=left_ast, comparators=comp_ast, ops=g.get(n, 'ops'), lineno=0, col_offset=0)

            return ast_Comp
        raise TypeError(f'Cannot convert {n},{na}')
    except:
        print(n)
        raise


def process_assign_node(target_nodes, g, var_def, value_ast, na, targets):
    if len(target_nodes) > 1:
        target_ast = []
        for target_node in target_nodes:
            target_ast.append(node_to_ast(target_node[1], g, var_def, read=False))
            targets.append(target_node[1])
        ast_assign = ast.Assign(targets=[ast.Tuple(elts=target_ast, lineno=0, col_offset=0, ctx=ast.Store())],
                                value=value_ast, lineno=0, col_offset=0)
        return ast_assign
    else:
        target_node = target_nodes[0][1]
        target_ast = node_to_ast(target_node, g, var_def, read=False)
        if value_ast and target_ast:
            if na == ast.Assign or target_node not in targets:
                targets.append(target_node)
                ast_assign = ast.Assign(targets=[target_ast], value=value_ast, lineno=0, col_offset=0)
            else:
                ast_assign = ast.AugAssign(target=target_ast, value=value_ast, lineno=0, col_offset=0, op=ast.Add())
            return ast_assign


def function_from_graph_generic(g: Graph, name, var_def_):
    lineno_count = 1
    decorators = []
    top_nodes = g.topological_nodes()

    var_def = var_def_.var_def

    body = []
    targets = []
    for n in top_nodes:
        lineno_count += 1

        if (at := g.get(n, 'ast_type')) == ast.Assign or at == ast.AugAssign:
            value_node = g.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.VALUE)[1][0][0]

            value_ast = node_to_ast(value_node, g, var_def, ctxread=True)
            body.append(process_assign_node(g.get_edges_for_node_filter(start_node=n, attr='e_type',
                                                                        val=EdgeType.TARGET)[1],
                                            g, var_def, value_ast, at, targets))
    var_def_.order_variables(g.arg_metadata)
    if (l := len(var_def_.get_targets())) > 1:
        return_ = ast.Return(value=ast.Tuple(elts=var_def_.get_order_trgs()))
    elif l == 1:
        return_ = ast.Return(value=var_def_.get_order_trgs()[0])
    else:
        g.as_graphviz('noret', force=True)
        raise IndexError(f'Function {name} should have return, no?')
    body.append(return_)
    args = ast.arguments(posonlyargs=[], args=var_def_.get_order_args(), vararg=None, defaults=[], kwonlyargs=[],
                         kwarg=None)

    func = wrap_function(name, body, decorators=decorators, args=args)

    target_ids = []
    for i, arg in enumerate(var_def_.args_order):
        if arg in var_def_.targets:
            target_ids.append(i)

    return func, var_def_.args_order, target_ids


def compiled_function_from_graph_generic_llvm(g: Graph, name, var_def_, imports,
                                              compiled_function=False):
    func, signature, fname, r_args, r_targets = function_from_graph_generic_llvm(g, name, var_def_)
    if not compiled_function:
        return func, signature, r_args, r_targets

    body = []
    for (module, name) in imports.as_imports:
        body.append(ast.Import(names=[ast.alias(name=module, asname=name)], lineno=0, col_offset=0, level=0))
    for (module, name) in imports.from_imports:
        body.append(
            ast.ImportFrom(module=module, names=[ast.alias(name=name, asname=None)], lineno=0, col_offset=0, level=0))
    body.append(func)
    body.append(ast.Return(value=ast.Name(id=fname, ctx=ast.Load(), lineno=0, col_offset=0), lineno=0, col_offset=0))

    func = wrap_function(fname + '1', body, decorators=[],
                         args=ast.arguments(posonlyargs=[], args=[], vararg=None, defaults=[],
                                            kwonlyargs=[], kw_defaults=[], kwarg=None))
    module_func = ast.Module(body=[func], type_ignores=[])
    code = compile(module_func, filename='llvm_equations_storage', mode='exec')
    namespace = {}
    exec(code, namespace)
    compiled_func = list(namespace.values())[1]()

    return compiled_func, signature, r_args, r_targets


def function_from_graph_generic_llvm(g: Graph, name, var_def_):
    fname = name + '_llvm'

    lineno_count = 1

    top_nodes = g.topological_nodes()

    var_def = var_def_.var_def

    body = []
    targets = []
    for n in top_nodes:
        lineno_count += 1

        if (na := g.get(n, 'ast_type')) == ast.Assign or na == ast.AugAssign:
            value_node = g.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.VALUE)[1][0][0]
            value_ast = node_to_ast(value_node, g, var_def, ctxread=True)

            body.append(
                process_assign_node(g.get_edges_for_node_filter(start_node=n, attr='e_type', val=EdgeType.TARGET)[1], g,
                                    var_def, value_ast, na, targets))

    var_def_.order_variables(g.arg_metadata)
    args = ast.arguments(posonlyargs=[], args=var_def_.get_order_args(), vararg=None, defaults=[],
                         kwonlyargs=[], kw_defaults=[], kwarg=None)
    signature = [f'void(']
    target_ids = []
    for i, arg in enumerate(var_def_.args_order):
        if arg in var_def_.targets:
            signature.append("CPointer(float64), ")
            target_ids.append(i)
        else:
            signature.append("float64, ")
    signature[-1] = signature[-1][:-2]
    signature.append(")")
    signature = ''.join(signature)
    decorators = []

    func = wrap_function(fname, body, decorators=decorators, args=args)
    return func, signature, fname, var_def_.args_order, target_ids


def postfix_from_branches(branches: dict):
    postfix = []
    for b, bv in branches.items():
        postfix += [b, str(bv)]
    return "_".join(postfix)


def parse_assign(value, target, ao, name, file, ln, g, tag_vars, prefix, branches):
    m, start = parse_(value, name, file, ln, g, tag_vars, prefix, branches=branches)
    mapped, end = parse_(target, name, file, ln, g, tag_vars, prefix, branches=branches)

    en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label='+=' if mapped else '=',
                    ast_type=ast.AugAssign if mapped else ast.Assign, node_type=NodeTypes.ASSIGN,
                    ast_op=ast.Add() if mapped else None))
    g.add_edge(start=en, end=end, e_type=EdgeType.TARGET, branches=branches.copy())
    if isinstance(start, list):
        for s in start:
            g.add_edge(start=s[0], end=en, e_type=EdgeType.VALUE, branches=s[1])
    else:
        g.add_edge(start=start, end=en, e_type=EdgeType.VALUE, branches=branches.copy())
    return en


def parse_(ao, name, file, ln, g: Graph, tag_vars, prefix='.', branches={}):
    en = None
    is_mapped = None

    if isinstance(ao, ast.Module):
        for b in ao.body:

            # Check if function def
            if isinstance(b, ast.FunctionDef):
                # Get name of function

                # Parse function
                for b_ in b.body:
                    parse_(b_, name, file, ln, g, tag_vars, prefix, branches)

    elif isinstance(ao, ast.Assign):

        # Check if attribute
        if isinstance(ao.targets[0], ast.Attribute) or isinstance(ao.targets[0], ast.Name) or isinstance(ao.targets[0],
                                                                                                         ast.Tuple):
            pass
        else:
            raise AttributeError('Unknown type of target: ', type(ao.targets[0]))
        if isinstance(ao.targets[0], ast.Tuple) and isinstance(ao.value, ast.Tuple):
            for i, _ in enumerate(ao.value.elts):
                en = parse_assign(ao.value.elts[i], ao.targets[0].elts[i], ao, name, file, ln, g, tag_vars, prefix,
                                  branches)
        elif isinstance(ao.targets[0], ast.Tuple):
            if isinstance(ao.value, ast.Call):
                m, start = parse_(ao.value, name, file, ln, g, tag_vars, prefix, branches=branches)
                mapped = False
                en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label='+=' if mapped else '=',
                                ast_type=ast.AugAssign if mapped else ast.Assign, node_type=NodeTypes.ASSIGNTUPLE,
                                ast_op=ast.Add() if mapped else None))
                g.add_edge(Edge(start=start, end=en, e_type=EdgeType.VALUE, branches=branches.copy()))
                for sa in ao.targets[0].elts:
                    mapped, end = parse_(sa, name, file, ln, g, tag_vars, prefix, branches=branches)
                    g.add_edge(Edge(start=en, end=end, e_type=EdgeType.TARGET, branches=branches.copy()))
            else:
                raise AttributeError('Assigning to tuple is not supported: ', type(ao.targets[0]))
        elif isinstance(ao.value, ast.Tuple):
            mapped, end = parse_(ao.targets[0], name, file, ln, g, tag_vars, prefix, branches=branches)

            en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label='+=' if mapped else '=',
                            ast_type=ast.AugAssign if mapped else ast.Assign, node_type=NodeTypes.ASSIGN,
                            ast_op=ast.Add() if mapped else None))

            g.add_edge(Edge(start=en, end=end, e_type=EdgeType.TARGET, branches=branches.copy()))
            for sa in ao.value.elts:
                m, start = parse_(sa, name, file, ln, g, tag_vars, prefix, branches=branches)
                g.add_edge(Edge(start=start, end=en, e_type=EdgeType.VALUE, branches=branches.copy()))
        else:
            en = parse_assign(ao.value, ao.targets[0], ao, name, file, ln, g, tag_vars, prefix, branches)

    elif isinstance(ao, ast.Num):
        # Constant
        source_id = 'c' + str(ao.value)
        en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label=source_id, ast_type=ast.Num, value=ao.value,
                        node_type=NodeTypes.VAR))

        # Check if simple name
    elif isinstance(ao, ast.Name) or isinstance(ao, ast.Attribute):
        local_id = recurse_Attribute(ao)

        source_id = local_id
        if source_id[:6] == 'scope.':
            scope_var = tag_vars[source_id[6:]]
            tag_vars[source_id[6:]].used_in_equation_graph = True

        else:
            scope_var = None

        if '-' in source_id:
            raise ValueError(f'Bad character -')

        node_type = NodeTypes.VAR

        if scope_var:

            var_type = VariableType.DERIVATIVE
            is_mapped = scope_var.sum_mapping or scope_var.mapping

        else:
            var_type = VariableType.PARAMETER

        en = g.add_node(Node(key=source_id, ao=ao, file=file, name=name, ln=ln, id=source_id, local_id=local_id,
                        ast_type=type(ao), node_type=node_type, scope_var=scope_var), ignore_existing=True)


    elif isinstance(ao, ast.UnaryOp):
        # Unary op
        op_sym = get_op_sym(ao.op)

        en = g.add_node(Node(label=op_sym, ast_type=ast.UnaryOp, node_type=NodeTypes.OP, ast_op=ao.op),
                        ignore_existing=True)

        m, start = parse_(ao.operand, name, file, ln, g, tag_vars, prefix, branches=branches)
        g.add_edge(Edge(start=start, e_type=EdgeType.OPERAND, end=en, branches=branches.copy()))

    elif isinstance(ao, ast.Call):

        op_name = recurse_Attribute(ao.func, sep='.')

        en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label=op_name, func=ao.func, ast_type=ast.Call,
                        node_type=NodeTypes.OP))

        for sa in ao.args:
            m, start = parse_(sa, name, file, ln, g, tag_vars, prefix=prefix, branches=branches)
            g.add_edge(Edge(start=start, end=en, e_type=EdgeType.ARGUMENT, branches=branches.copy()))


    elif isinstance(ao, ast.BinOp):

        op_sym = get_op_sym(ao.op)
        en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label=op_sym, ast_type=ast.BinOp,
                        node_type=NodeTypes.OP, ast_op=ao.op))

        for a in ['left', 'right']:
            m, start = parse_(getattr(ao, a), name, file, ln, g, tag_vars, prefix, branches=branches)
            g.add_edge(Edge(start=start, end=en, e_type=str_to_edgetype(a), branches=branches.copy()))

    elif isinstance(ao, ast.Compare):
        ops_sym = [get_op_sym(o) for o in ao.ops]

        en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label=''.join(ops_sym), ast_type=ast.Compare,
                        node_type=NodeTypes.OP, ops=ao.ops))

        m, start = parse_(ao.left, name, file, ln, g, tag_vars, prefix=prefix, branches=branches)

        g.add_edge(Edge(start=start, end=en, label=f'left', e_type=EdgeType.LEFT, branches=branches))

        for i, sa in enumerate(ao.comparators):
            m, start = parse_(sa, name, file, ln, g, tag_vars, prefix=prefix, branches=branches)
            g.add_edge(Edge(start=start, end=en, label=f'comp{i}', e_type=EdgeType.COMP, branches=branches))

    elif isinstance(ao, ast.If):
        new_branch = None
        if isinstance(ao.test, ast.Attribute):
            source_id = recurse_Attribute(ao.test)

            if source_id[:6] == 'scope.':
                scope_var = tag_vars[source_id[6:]]
                tag_vars[source_id[6:]].used_in_equation_graph = True

                if scope_var.type == VariableType.CONSTANT:
                    new_branch = scope_var.tag
                    branches_t = deepcopy(branches)
                    branches_t[new_branch] = True
                    m_t, start_t = parse_(getattr(ao, 'body'), name, file, ln, g, tag_vars, prefix, branches=branches_t)

                    branches_f = deepcopy(branches)
                    branches_f[new_branch] = False

                    m_f, start_f = parse_(getattr(ao, 'orelse'), name, file, ln, g, tag_vars, prefix,
                                          branches=branches_f)

                    return [m_t, m_f], [(start_t, branches_t), (start_f, branches_f)]

        en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label='if_st',
                             ast_type=ast.If, node_type=NodeTypes.OP))
        for a in ['body', 'orelse', 'test']:
            if isinstance(getattr(ao, a), list):
                for a_ in getattr(ao, a):
                    m, start = parse_(a_, name, file, ln, g, tag_vars, prefix, branches=branches)
                    g.add_edge(Edge(start=start, end=en, e_type=str_to_edgetype(a), branches=branches))
            else:
                m, start = parse_(a, name, file, ln, g, tag_vars, prefix, branches=branches)
                g.add_edge(Edge(start=start, end=en, e_type=str_to_edgetype(a), branches=branches))

    elif isinstance(ao, ast.IfExp):
        if isinstance(ao.test, ast.Attribute):
            source_id = recurse_Attribute(ao.test)

            if source_id[:6] == 'scope.':
                scope_var = tag_vars[source_id[6:]]
                tag_vars[source_id[6:]].used_in_equation_graph = True

                if scope_var.type == VariableType.CONSTANT:
                    new_branch = scope_var.tag
                    branches_t = deepcopy(branches)
                    branches_t[new_branch] = True
                    m_t, start_t = parse_(getattr(ao, 'body'), name, file, ln, g, tag_vars, prefix, branches=branches_t)

                    branches_f = deepcopy(branches)
                    branches_f[new_branch] = False

                    m_f, start_f = parse_(getattr(ao, 'orelse'), name, file, ln, g, tag_vars, prefix,
                                          branches=branches_f)

                    return [m_t, m_f], [(start_t, branches_t), (start_f, branches_f)]

        en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label='if_exp',
                             ast_type=ast.IfExp, node_type=NodeTypes.OP))
        for a in ['body', 'orelse', 'test']:
            m, start = parse_(getattr(ao, a), name, file, ln, g, tag_vars, prefix, branches=branches)

            g.add_edge(Edge(start=start, end=en, e_type=str_to_edgetype(a), branches=branches))

    else:
        raise TypeError('Cannot parse <' + str(type(ao)) + '>')

    return is_mapped, en


def qualify(s, prefix):
    return prefix + '.' + s.replace('scope.', '')


def qualify_equation(prefix, g, tag_vars):
    def q(s):
        return qualify(s, prefix)

    g_qual = g.clone()
    # update keys
    g_qual.node_map = {q(k): v for k, v in g_qual.node_map.items()}
    g_qual.key_map = {k: q(v) for k, v in g_qual.key_map.items()}

    scope_vars_qual = [tag_vars[sv.tag] if isinstance(sv := g.get(n, 'scope_var'), ScopeVariable) else sv
                       for n in g.node_map.values()]

    for i in range(g_qual.node_counter):
        g_qual.nodes[i].scope_var = scope_vars_qual[i]

    return g_qual


def parse_eq(model_namespace, item_id, equation_graph: Graph, nodes_dep, scope_variables,
             parsed_eq_branches, scoped_equations, parsed_eq):
    for m in model_namespace.equation_dict.values():
        for eq in m:
            is_set = model_namespace.is_set
            if is_set:
                eq_key = "EQ_SET" + eq.id.replace(".", "_").replace("-", "_")
            else:
                eq_key = "EQ_" + eq.id.replace(".", "_").replace("-", "_")
            is_parsed_eq = eq_key in parsed_eq
            if not is_parsed_eq:
                dsource = eq.lines

                tries = 0
                while tries < 10:
                    try:
                        dsource = dedent(dsource)
                        ast_tree = ast.parse(dsource)
                        break
                    except IndentationError:

                        tries += 1
                        if tries > 10 - 1:
                            print(dsource)
                            raise

                g = Graph()
                branches = {}
                parse_(ast_tree, eq_key, eq.file, eq.lineno, g, scope_variables, branches=branches)

                # Create branched versions of graph

                branches_ = set()
                [branches_.update(b.keys()) for b.branches in g.edges_c[:g.edge_counter] if b.branches]
                all_branches = [{}]
                from copy import deepcopy
                for b in branches_:

                    for a in all_branches:
                        a.update({b: True})

                    all_branches += deepcopy(all_branches)
                    for a in all_branches[int(len(all_branches) / 2):]:
                        a[b] = False

                if len(all_branches) > 1:
                    branch_graphs = []
                    for a in all_branches:

                        gb = g.clone()

                        for i, b in enumerate(gb.edges_attr['branches'][:g.edge_counter]):

                            for ak in a.keys():
                                if ak in b and b[ak] != a[ak]:
                                    gb.remove_edge(i)

                        gb = gb.clean()
                        branch_graphs.append((a, gb, eq_key + '_' + postfix_from_branches(a)))

                    for branch in branch_graphs:
                        parsed_eq_branches[branch[2]] = (eq, dsource, branch[1], branch[0])

                else:
                    parsed_eq_branches[eq_key] = (eq, dsource, g, {})

                parsed_eq[eq_key] = list(branches_)

            g = parsed_eq_branches[eq_key][2]

            ns_path = model_namespace.full_tag
            eq_path = ns_path + '.' + eq_key
            g_qualified = qualify_equation(ns_path, g, scope_variables)

            # make equation graph
            eq_name = ('EQ_' + eq_path).replace('.', '_')

            scoped_equations[eq_name] = eq_key

            eq_n = equation_graph.add_node(Node(key=eq_name,
                                           node_type=NodeTypes.EQUATION,
                                           name=eq_name, file=eq_name, ln=0, label=eq_name,
                                           ast_type=ast.Call,
                                           vectorized=is_set,
                                           item_id=item_id,
                                           func=ast.Name(id=eq_key.replace('.', '_'))))

            for n in range(g_qualified.node_counter):

                if g_qualified.get(n, attr='node_type') == NodeTypes.VAR and g_qualified.get(n, attr='scope_var'):

                    n_key = g_qualified.key_map[n]

                    if not n_key in nodes_dep:
                        nodes_dep[n_key] = []
                    if not eq_name in nodes_dep[n_key]:
                        nodes_dep[n_key].append(eq_name)

                    sv = g_qualified.get(n, 'scope_var')
                    neq = equation_graph.add_node(Node(key=sv.id, node_type=NodeTypes.VAR, scope_var=sv,
                                                   is_set_var=is_set, label=sv.get_path_dot()), ignore_existing=True)

                    targeted = False
                    read = False

                    end_edges = g_qualified.get_edges_for_node(end_node=n)

                    try:
                        next(end_edges)
                        equation_graph.add_edge(Edge(start=eq_n, end=neq, e_type=EdgeType.TARGET, arg_local=sv.id if sv else 'local'))
                        targeted = True
                    except StopIteration:
                        pass

                    if not targeted and not read:
                        start_edges = g_qualified.get_edges_for_node(start_node=n)
                        try:
                            next(start_edges)
                            equation_graph.add_edge(Edge(neq, eq_n, e_type=EdgeType.ARGUMENT, arg_local=sv.id if (
                                sv := g_qualified.get(n, 'scope_var')) else 'local'))
                        except StopIteration:
                            pass
            if not is_parsed_eq:
                for sv in scope_variables:
                    if scope_variables[sv].used_in_equation_graph:
                        g.arg_metadata.append((sv, scope_variables[sv].id, scope_variables[sv].used_in_equation_graph))
                        scope_variables[sv].used_in_equation_graph = False
                    else:
                        g.arg_metadata.append((sv, scope_variables[sv].id, scope_variables[sv].used_in_equation_graph))


def process_mappings(mappings, equation_graph: Graph, nodes_dep, scope_vars):
    logging.info('process mappings')
    for m in mappings:
        target_var = scope_vars[m[0]]
        target_set_var_ix = -1
        if target_var.set_var:
            target_set_var_ix = target_var.set_var_ix
            target_var = target_var.set_var
        target_var_id = target_var.id

        node_type = NodeTypes.VAR

        t = equation_graph.add_node(Node(key=target_var_id, file='mapping', name=m, ln=0, id=target_var_id,
                                    label=target_var.get_path_dot(), ast_type=ast.Attribute, node_type=node_type,
                                    scope_var=target_var,  set_var_ix=target_set_var_ix),ignore_existing=False,)

        if not target_var_id in nodes_dep:
            nodes_dep[target_var_id] = []

        for i in m[1]:

            ivar_var = scope_vars[i]
            ivar_set_var_ix = -1

            if ivar_var.set_var:
                ivar_set_var_ix = ivar_var.set_var_ix
                ivar_var = ivar_var.set_var

            ivar_id = ivar_var.id

            if not ivar_id in nodes_dep:
                nodes_dep[ivar_id] = []

            ivar_node_e = equation_graph.add_node(Node(key=ivar_id, file='mapping', name=m, ln=0, id=ivar_id,
                                                  label=ivar_var.get_path_dot(),
                                                  ast_type=ast.Attribute, node_type=NodeTypes.VAR, scope_var=ivar_var,
                                                  set_var_ix=ivar_set_var_ix), ignore_existing=False)

            ix_ = equation_graph.has_edge_for_nodes(start_node=ivar_node_e, end_node=t)
            lix = len(ix_)
            if lix == 0:
                equation_graph.add_edge(Edge(ivar_node_e, t, e_type=EdgeType.MAPPING,
                                        mappings=[(ivar_set_var_ix, target_set_var_ix)]))
            else:
                equation_graph.edges_attr['mappings'][ix_[0]].append((ivar_set_var_ix, target_set_var_ix))

    logging.info('clone eq graph')
    equation_graph = equation_graph.clean()

    logging.info('remove dependencies')

    logging.info('cleaning')

    logging.info('done cleaning')

    return equation_graph
