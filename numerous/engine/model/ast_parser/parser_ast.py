import ast
from textwrap import dedent

from numerous.engine.variables import Variable
from numerous.engine.model.ast_parser.ast_visitor import ast_to_graph, connect_equation_node
from numerous.engine.model.graph_representation import Graph, EdgeType, Node, Edge
from numerous.engine.model.utils import NodeTypes
from numerous.utils import logger as log
from copy import deepcopy


class ParsedEquation:
    def __init__(self, eq, dsource, graph, t, is_set, replacements=None):
        self.eq = eq
        self.is_set = is_set
        self.dsource = dsource
        self.graph = graph
        self.t = t
        self.replacements = replacements


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


def qualify_equation(prefix, g, tag_vars, eq_class, eq_current):
    def q(s):
        return qualify(s, prefix)

    g_qual = g.clone()
    # update keys
    g_qual.node_map = {q(k): v for k, v in g_qual.node_map.items()}
    g_qual.key_map = {k: q(v) for k, v in g_qual.key_map.items()}

    scope_vars_qual = [tag_vars[sv.tag] if isinstance(sv := g.nodes[n].scope_var, Variable) else sv
                       for n in g.node_map.values()]

    for i in range(g_qual.node_counter):
        g_qual.nodes[i].scope_var = scope_vars_qual[i]

    replacements = {}
    refer_to_self = False
    import inspect, hashlib, json

    closure_vars = inspect.getclosurevars(eq_class)

    func_ = closure_vars.nonlocals['func']
    closure_vars_func = inspect.getclosurevars(func_)
    replacements.update(closure_vars_func.globals)

    for n in g.nodes:

        if (f := n.func) is not None and hasattr(f, 'value'):
            if f.value.id == 'self':
                refer_to_self = True
                obj = getattr(eq_class.__self__, f.attr)
                replacements[n.id] = obj

    eq_key = 'var_' + eq_current

    return g_qual, refer_to_self, eq_key, replacements


def _generate_equation_key(equation_id: str, is_set: bool) -> str:
    if is_set:
        eq_key = "EQ_SET" + equation_id.replace(".", "_").replace("-", "_")
    else:
        eq_key = "EQ_" + equation_id.replace(".", "_").replace("-", "_")
    return eq_key


def replace_global_var_identification(dsource):
    return dsource.replace("global_vars.", "global_vars_")


def parse_eq(model_namespace, item_id, mappings_graph: Graph, variables,
             parsed_eq_branches, eq_used):
    for m in model_namespace.equation_dict.values():
        for eq in m:
            ns_path = model_namespace.full_tag
            is_set = model_namespace.is_set
            eq_key = _generate_equation_key(eq.id, is_set)
            is_parsed_eq = eq_key in eq_used
            ast_tree = None
            if not is_parsed_eq:
                dsource = eq.lines
                dsource = replace_global_var_identification(dsource)
                tries = 0
                while tries < 10:
                    try:
                        dsource = dedent(dsource)
                        ast_tree = ast.parse(dsource)
                        break
                    except IndentationError:

                        tries += 1
                        if tries > 10 - 1:
                            raise

                g = ast_to_graph(ast_tree, eq_key, eq.file, eq.lineno, variables)
                # Create branched versions of graph
                branches_ = set()
                [branches_.update(b.branches.keys()) for b in g.edges_c[:g.edge_counter] if b.branches]
                all_branches = [{}]
                for b in branches_:

                    for a in all_branches:
                        a.update({b: True})

                    all_branches += deepcopy(all_branches)
                    for a in all_branches[int(len(all_branches) / 2):]:
                        a[b] = False

                if len(all_branches) > 1:
                    branch_graphs = []
                    for a in all_branches:

                        gb = g._clone()

                        for i, b in enumerate(gb.edges_attr['branches'][:g.edge_counter]):

                            for ak in a.keys():
                                if ak in b and b[ak] != a[ak]:
                                    gb.remove_edge(i)

                        gb = gb.clean()
                        branch_graphs.append((a, gb, eq_key + '_' + postfix_from_branches(a)))

                    for branch in branch_graphs:
                        parsed_eq_branches[branch[2]] = ParsedEquation(eq, dsource, branch[1], branch[0], is_set)

                else:
                    parsed_eq_branches[eq_key] = ParsedEquation(eq, dsource, g, {}, is_set)

            g = parsed_eq_branches[eq_key].graph

            g_qualified, refer_to_self, eq_key_, replacements = qualify_equation(ns_path, g, variables, eq,
                                                                                 eq_key)

            if len(replacements) > 0:
                if refer_to_self:
                    eq_key__ = eq_key_
                else:
                    eq_key__ = eq_key

                parsed_eq_branches[eq_key__] = ParsedEquation(
                    parsed_eq_branches[eq_key].eq, parsed_eq_branches[eq_key].dsource, parsed_eq_branches[eq_key].graph,
                    {}, is_set, replacements)

                eq_key = eq_key__

            eq_used.append(eq_key)

            # make equation graph

            node = Node(key=eq_key + "_" + ns_path,
                        node_type=NodeTypes.EQUATION,
                        name=eq_key, file=eq_key, ln=0, label=eq_key,
                        ast_type=ast.Call,
                        vectorized=is_set,
                        item_id=item_id,
                        func=ast.Name(id=eq_key.replace('.', '_')))

            connect_equation_node(g_qualified, mappings_graph, node, is_set)

            if not is_parsed_eq:
                for sv in variables:
                    if variables[sv].used_in_equation_graph:
                        variables[sv].add_eq_used(eq_key)
                        g.arg_metadata.append(variables[sv])
                        variables[sv].used_in_equation_graph = False
                    else:
                        g.arg_metadata.append(variables[sv])


def process_mappings(mappings, mappings_graph: Graph, scope_vars):
    for m in mappings:
        target_var = scope_vars[m[0]]
        target_set_var_ix = -1
        if target_var.set_var:
            target_set_var_ix = target_var.set_var_ix
            target_var = target_var.set_var
        target_var_id = target_var.id

        node_type = NodeTypes.VAR

        t = mappings_graph.add_node(Node(key=target_var_id, file='mapping', name=m, ln=0, id=target_var_id,
                                         label=target_var.get_path_dot(), ast_type=ast.Attribute, node_type=node_type,
                                         scope_var=target_var, set_var_ix=target_set_var_ix), ignore_existing=False, )

        for i in m[1]:

            ivar_var = scope_vars[i]
            ivar_set_var_ix = -1

            if ivar_var.set_var:
                ivar_set_var_ix = ivar_var.set_var_ix
                ivar_var = ivar_var.set_var

            ivar_id = ivar_var.id

            ivar_node_e = mappings_graph.add_node(Node(key=ivar_id, file='mapping', name=m, ln=0, id=ivar_id,
                                                       label=ivar_var.get_path_dot(),
                                                       ast_type=ast.Attribute, node_type=NodeTypes.VAR,
                                                       scope_var=ivar_var,
                                                       set_var_ix=ivar_set_var_ix), ignore_existing=False)

            ix_ = mappings_graph.has_edge_for_nodes(start_node=ivar_node_e, end_node=t)
            lix = len(ix_)
            if lix == 0:
                mappings_graph.add_edge(Edge(ivar_node_e, t, e_type=EdgeType.MAPPING,
                                             mappings=[(ivar_set_var_ix, target_set_var_ix)]))
            else:
                mappings_graph.edges_c[ix_[0]].mappings.append((ivar_set_var_ix, target_set_var_ix))

    log.info('Clone eq graph')

    log.info('Remove dependencies')

    log.info('Cleaning')

    mappings_graph = mappings_graph.clean()

    log.info('Cleaning finished')
    return mappings_graph
