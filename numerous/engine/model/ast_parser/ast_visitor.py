import ast
from copy import deepcopy

from numerous.engine.model.graph_representation.graph import Node, Edge, Graph
from numerous.engine.model.utils import NodeTypes, recurse_Attribute
from numerous.engine.variables import VariableType
from numerous.engine.model.graph_representation.utils import str_to_edgetype, EdgeType

op_sym_map = {ast.Add: '+', ast.Sub: '-', ast.Div: '/', ast.Mult: '*', ast.Pow: '**', ast.USub: '*-1',
              ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}


def get_op_sym(op):
    return op_sym_map[type(op)]


def parse_assign(value, target, ao, name, file, ln, g, tag_vars, prefix, branches):
    m, start = parse_(value, name, file, ln, g, tag_vars, prefix, branches=branches)
    mapped, end = parse_(target, name, file, ln, g, tag_vars, prefix, branches=branches)

    en = g.add_node(Node(ao=ao, file=file, name=name, ln=ln, label='+=' if mapped else '=',
                         ast_type=ast.AugAssign if mapped else ast.Assign, node_type=NodeTypes.ASSIGN,
                         ast_op=ast.Add() if mapped else None))
    g.add_edge(Edge(start=en, end=end, e_type=EdgeType.TARGET, branches=branches.copy()))

    if isinstance(start, list):
        for s in start:
            g.add_edge(Edge(start=s[0], end=en, e_type=EdgeType.VALUE, branches=s[1]))
    else:
        g.add_edge(Edge(start=start, end=en, e_type=EdgeType.VALUE, branches=branches.copy()))
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


def ast_to_graph(ast_tree, eq_key, eq, scope_variables):
    g = Graph()
    branches = {}
    parse_(ast_tree, eq_key, eq.file, eq.lineno, g, scope_variables, branches=branches)
    return g
