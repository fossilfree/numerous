import ast
import logging
from textwrap import dedent

from ast_visitor import ast_to_graph
from numerous.engine.model.graph_representation import Graph, EdgeType, Node, Edge
from numerous.engine.model.utils import NodeTypes
from numerous.engine.scope import ScopeVariable


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


def postfix_from_branches(branches: dict):
    postfix = []
    for b, bv in branches.items():
        postfix += [b, str(bv)]
    return "_".join(postfix)


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


def _generate_equation_key(equation_id: str, is_set: bool) -> str:
    if is_set:
        eq_key = "EQ_SET" + equation_id.replace(".", "_").replace("-", "_")
    else:
        eq_key = "EQ_" + equation_id.replace(".", "_").replace("-", "_")
    return eq_key


def parse_eq(model_namespace, item_id, equation_graph: Graph, nodes_dep, scope_variables,
             parsed_eq_branches, scoped_equations, parsed_eq):
    for m in model_namespace.equation_dict.values():
        for eq in m:
            ns_path = model_namespace.full_tag
            is_set = model_namespace.is_set
            eq_key = _generate_equation_key(eq.id, is_set)
            is_parsed_eq = eq_key in parsed_eq
            ast_tree = None
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

                g = ast_to_graph(ast_tree, eq_key, eq, scope_variables)
                # Create branched versions of graph
                branches_ = set()
                [branches_.update(b.branches.keys()) for b in g.edges_c[:g.edge_counter] if b.branches]
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

                parsed_eq[eq_key] = 'EQ_' + ns_path + '.' + eq.name

            g = parsed_eq_branches[eq_key][2]

            eq_path = ns_path + '.' + eq_key
            g_qualified = qualify_equation(ns_path, g, scope_variables)

            # make equation graph
            eq_name = ('EQ_' + eq_path).replace('.', '_')

            scoped_equations[eq_name] = eq_key

            eq_n = equation_graph.add_node(Node(key=eq_name,
                                                node_type=NodeTypes.EQUATION,
                                                name=eq_name, file=eq_name, ln=0, label=parsed_eq[eq_key],
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
                                                       is_set_var=is_set, label=sv.get_path_dot()),
                                                  ignore_existing=True)

                    targeted = False
                    read = False

                    end_edges = g_qualified.get_edges_for_node(end_node=n)

                    try:
                        next(end_edges)
                        equation_graph.add_edge(
                            Edge(start=eq_n, end=neq, e_type=EdgeType.TARGET, arg_local=sv.id if sv else 'local'))
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
                                         scope_var=target_var, set_var_ix=target_set_var_ix), ignore_existing=False, )

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
                                                       ast_type=ast.Attribute, node_type=NodeTypes.VAR,
                                                       scope_var=ivar_var,
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
