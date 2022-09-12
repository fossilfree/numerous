import ast
from itertools import chain, zip_longest
from typing import Any

from numerous.engine.model.graph_representation.graph import Node, Edge, Graph
from numerous.engine.model.graph_representation.utils import str_to_edgetype, EdgeType
from numerous.engine.model.utils import NodeTypes, recurse_Attribute

op_sym_map = {ast.Add: '+', ast.Sub: '-', ast.Div: '/', ast.Mult: '*', ast.Pow: '**', ast.USub: '*-1',
              ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>', ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!='}


def get_op_sym(op):
    return op_sym_map[type(op)]


def ast_to_graph(ast_tree, eq_key, eq_file, eq_lineno, scope_variables):
    parser = AstVisitor(eq_key, eq_file, eq_lineno, scope_variables)
    parser.visit(ast_tree)
    return parser.graph


class AstVisitor(ast.NodeVisitor):

    def __init__(self, eq_key, eq_file, eq_lineno, scope_variables):
        self.graph = Graph(label=eq_key)
        self.CONSTANT_LABEL = 'c'
        self.SCOPE_LABEL = 'scope.'
        self.IFEXPATTRIBUTES = ['body', 'orelse', 'test']
        self.branches = []
        self.eq_key = eq_key
        self.eq_file = eq_file
        self.eq_lineno = eq_lineno
        self.mapped_stack = []
        self.node_number_stack = []
        self.scope_variables = scope_variables
        self._supported_assign_target = (ast.Attribute, ast.Name, ast.Tuple)

    def traverse(self, node: ast.AST):
        if isinstance(node, list):
            for item in node:
                self.traverse(item)
        else:
            super().visit(node)

    def visit(self, node: ast.AST):
        self.traverse(node)

    def visit_Module(self, node: ast.Module):
        self.traverse(node.body)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.traverse(node.body)

    def visit_Call(self, node: ast.Call) -> Any:
        self._process_named_node(node.func, ast_type=ast.Call, node_type=NodeTypes.OP, func=node.func)
        self.mapped_stack.pop()
        en = self.node_number_stack.pop()
        for sa in node.args:
            self.traverse(sa)
            start = self.node_number_stack.pop()
            self.mapped_stack.pop()
            self.graph.add_edge(Edge(start=start[0], end=en[0],
                                     e_type=EdgeType.ARGUMENT, branches=self.branches.copy()))
        self.mapped_stack.append([None])
        self.node_number_stack.append(en)

    def visit_List(self, node: ast.List) -> Any:
        self._visit_iterable(node=node, ast_type=ast.List, ctx=node.ctx)

    def visit_Set(self, node: ast.Set) -> Any:
        self._visit_iterable(node=node, ast_type=ast.Set)

# TODO rewrite tuple as set and list
    def visit_Tuple(self, node: ast.Tuple):
        en = []
        mapped = []
        for el in node.elts:
            self.traverse(el)
            en.append(self.node_number_stack.pop())
            mapped.append(self.mapped_stack.pop())
        self.mapped_stack.append(list(chain.from_iterable(mapped)))
        self.node_number_stack.append(list(chain.from_iterable(en)))

    def _visit_iterable(self, node, ast_type, ctx=None):
        en = self.graph.add_node(Node(ao=Node, file=self.eq_file,
                                      name=self.eq_key, ln=self.eq_lineno,
                                      label="iterable_node", ctx=ctx, node_type=NodeTypes.OP,
                                      ast_type=ast_type))

        for el in node.elts:
            self.traverse(el)
            start = self.node_number_stack.pop()
            self.graph.add_edge(Edge(start=start[0], end=en, e_type=EdgeType.ELEMENT, branches=self.branches.copy()))
        self.mapped_stack.append([None])
        self.node_number_stack.append([en])


    def _is_assign_target_supported(self, target: ast.expr):
        return isinstance(target, self._supported_assign_target)

    def _process_named_node(self, node: ast.expr, ast_type: type, node_type: NodeTypes, static_key=False, func=None):
        source_id = recurse_Attribute(node)
        scope_var, is_mapped = self._select_scope_var(source_id)
        en = self.graph.add_node(
            Node(key=source_id if static_key else None, ao=node, file=self.eq_file, name=self.eq_key,
                 ln=self.eq_lineno, id=source_id, local_id=source_id, func=func,
                 ast_type=ast_type, node_type=node_type, scope_var=scope_var),
            ignore_existing=True)
        self.mapped_stack.append([is_mapped])
        self.node_number_stack.append([en])

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        operation = node.op
        target = node.target
        value = node.value

        new_value = ast.BinOp()

        new_value.left = target
        new_value.right = value
        new_value.op = operation

        new_assign = ast.Assign()
        new_assign.targets = [target]
        new_assign.value = new_value

        self.visit_Assign(new_assign)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if self._is_assign_target_supported(target):
                self.traverse(target)
            else:
                raise AttributeError('Unknown type of assign target in Equation: ', type(target))
        self.traverse(node.value)
        self.mapped_stack.pop()
        mapped = self.mapped_stack.pop()
        start = self.node_number_stack.pop()
        end = self.node_number_stack.pop()
        en = 0
        for s, e, m in zip_longest(start, end, mapped):
            if e is not None and s is not None:
                en = self.graph.add_node(Node(ao=node, file=self.eq_file, name=self.eq_key, ln=self.eq_lineno,
                                              label='+=' if m else '=',
                                              ast_type=ast.AugAssign if m else ast.Assign, node_type=NodeTypes.ASSIGN,
                                              ast_op=ast.Add() if m else None))
                self.graph.add_edge(Edge(start=en, end=e, e_type=EdgeType.TARGET, branches=self.branches.copy()))
                self.graph.add_edge(Edge(start=s, end=en, e_type=EdgeType.VALUE, branches=self.branches.copy()))
            else:
                if e is not None:
                    self.graph.add_edge(Edge(start=en, end=e, e_type=EdgeType.TARGET, branches=self.branches.copy()))
                if s is not None:
                    self.graph.add_edge(Edge(start=s, end=en, e_type=EdgeType.VALUE, branches=self.branches.copy()))
        self.node_number_stack.append([en])
        self.mapped_stack.append(mapped)

    def visit_Attribute(self, node: ast.Attribute):
        self._process_named_node(node, ast_type=ast.Attribute, static_key=True, node_type=NodeTypes.VAR)

    def visit_Name(self, node: ast.Name):
        self._process_named_node(node, ast_type=ast.Name, static_key=True, node_type=NodeTypes.VAR)

    def _select_scope_var(self, source_id: str):
        scope_label_length = len(self.SCOPE_LABEL)
        if source_id[:scope_label_length] == self.SCOPE_LABEL:
            scope_var = self.scope_variables[source_id[scope_label_length:]]
            self.scope_variables[source_id[scope_label_length:]].used_in_equation_graph = True
            is_mapped = scope_var.sum_mapping or scope_var.mapping
        else:
            scope_var = None
            is_mapped = None
        return scope_var, is_mapped

    def visit_Constant(self, node: ast.Constant):
        source_id = self.CONSTANT_LABEL + str(node.value)
        en = self.graph.add_node(Node(ao=node, file=self.eq_file, name=self.eq_key, ln=self.eq_lineno,
                                      label=source_id, ast_type=ast.Constant, value=node.value,
                                      node_type=NodeTypes.VAR))
        self.mapped_stack.append([None])
        self.node_number_stack.append([en])

    def visit_Subscript(self, node: ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Slice):
            raise Exception('Slices are not supported right now')
        if isinstance(sl, ast.Tuple):
            for el in sl.elts:
                if isinstance(el, ast.Slice):
                    raise Exception('Slices are not supported right now')
        en = self.graph.add_node(Node(ao=Node, file=self.eq_file,
                                      name=self.eq_key, ln=self.eq_lineno,
                                      label="subscript", ctx=node.ctx, node_type=NodeTypes.SUBSCRIPT,
                                      ast_type=ast.Subscript))
        self.traverse(node.value)
        self.mapped_stack.pop()
        start = self.node_number_stack.pop()
        self.graph.add_edge(Edge(start=start[0], end=en,
                                 e_type=str_to_edgetype("subscript_value"), branches=self.branches.copy()))
        self.traverse(node.slice)
        mapped = self.mapped_stack.pop()
        start = self.node_number_stack.pop()
        self.graph.add_edge(Edge(start=start[0], end=en,
                                 e_type=str_to_edgetype("slice"), branches=self.branches.copy()))
        self.node_number_stack.append([en])
        self.mapped_stack.append(mapped)

    def visit_BinOp(self, node: ast.BinOp):
        op_sym = get_op_sym(node.op)
        en = self.graph.add_node(Node(ao=node, file=self.eq_file, name=self.eq_key,
                                      ln=self.eq_lineno, label=op_sym, ast_type=ast.BinOp,
                                      node_type=NodeTypes.OP, ast_op=node.op))

        self.traverse(node.left)
        self.mapped_stack.pop()
        start = self.node_number_stack.pop()
        self.graph.add_edge(Edge(start=start[0], end=en,
                                 e_type=str_to_edgetype("left"), branches=self.branches.copy()))
        self.traverse(node.right)
        mapped = self.mapped_stack.pop()
        start = self.node_number_stack.pop()
        self.graph.add_edge(Edge(start=start[0], end=en,
                                 e_type=str_to_edgetype("right"), branches=self.branches.copy()))
        self.node_number_stack.append([en])
        self.mapped_stack.append(mapped)

    def visit_UnaryOp(self, node: ast.UnaryOp):

        op_sym = get_op_sym(node.op)
        en = self.graph.add_node(Node(label=op_sym, ast_type=ast.UnaryOp, node_type=NodeTypes.OP, ast_op=node.op),
                                 ignore_existing=True)
        self.traverse(node.operand)
        start = self.node_number_stack.pop()
        self.graph.add_edge(Edge(start=start[0], e_type=EdgeType.OPERAND, end=en, branches=self.branches.copy()))
        self.node_number_stack.append([en])

    def visit_Compare(self, node: ast.Compare) -> Any:
        ops_sym = [get_op_sym(o) for o in node.ops]
        en = self.graph.add_node(Node(ao=node, file=self.eq_file, name=self.eq_key, ln=self.eq_lineno,
                                      label=''.join(ops_sym), ast_type=ast.Compare,
                                      node_type=NodeTypes.OP, ops=node.ops))
        self.traverse(node.left)
        start = self.node_number_stack.pop()
        self.mapped_stack.pop()
        self.graph.add_edge(Edge(start=start[0], end=en, label=f'left', e_type=EdgeType.LEFT, branches=self.branches))
        for i, sa in enumerate(node.comparators):
            self.traverse(sa)
            start = self.node_number_stack.pop()
            mapped = self.mapped_stack.pop()
            self.graph.add_edge(
                Edge(start=start[0], end=en, label=f'comp{i}', e_type=EdgeType.COMP, branches=self.branches))
        self.mapped_stack.append(mapped)
        self.node_number_stack.append([en])

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        en = self.graph.add_node(Node(ao=node, file=self.eq_file, name=self.eq_key, ln=self.eq_lineno, label='if_exp',
                                      ast_type=ast.IfExp, node_type=NodeTypes.OP))
        for a in self.IFEXPATTRIBUTES:
            self.traverse(getattr(node, a))
            start = self.node_number_stack.pop()
            mapped = self.mapped_stack.pop()
            self.graph.add_edge(Edge(start=start[0], end=en, e_type=str_to_edgetype(a), branches=self.branches))
        self.mapped_stack.append(mapped)
        self.node_number_stack.append([en])

    def visit_If(self, node: ast.If) -> Any:
        subgraph_body = ast_to_graph(node.body, self.eq_key, self.eq_file, self.eq_lineno, self.scope_variables)
        subgraph_test = ast_to_graph(node.test, self.eq_key, self.eq_file, self.eq_lineno, self.scope_variables)
        node = Node(ao=node, file=self.eq_file, name=self.eq_key,
                    ln=self.eq_lineno, label="IF", ast_type=ast.If, node_type=NodeTypes.IF, subgraph_body=subgraph_body,
                    subgraph_test=subgraph_test)
        subgraph_node_idx = connect_equation_node(subgraph_body, self.graph, node, is_set=False, include_local=True)
        connect_equation_node(subgraph_test, self.graph, node, is_set=False, include_local=True)
        self.mapped_stack.append([None])
        self.node_number_stack.append(subgraph_node_idx)


def connect_equation_node(equation_graph, mappings_graph, node, is_set, include_local=False):
    eq_node_idx = mappings_graph.add_node(node)
    for n in range(equation_graph.node_counter):
        if equation_graph.nodes[n].node_type == NodeTypes.VAR:
            if equation_graph.nodes[n].scope_var:
                sv = equation_graph.nodes[n].scope_var
                neq = mappings_graph.add_node(Node(key=sv.id, node_type=NodeTypes.VAR, scope_var=sv,
                                                   is_set_var=is_set, label=sv.get_path_dot()),
                                              ignore_existing=True)

                targeted = False
                read = False

                end_edges = equation_graph.get_edges_for_node(end_node=n)

                try:
                    next(end_edges)
                    mappings_graph.add_edge(
                        Edge(start=eq_node_idx, end=neq, e_type=EdgeType.TARGET, arg_local=sv.id))
                    targeted = True
                except StopIteration:
                    pass

                if not targeted and not read:
                    start_edges = equation_graph.get_edges_for_node(start_node=n)
                    try:
                        next(start_edges)
                        mappings_graph.add_edge(Edge(neq, eq_node_idx, e_type=EdgeType.ARGUMENT, arg_local=sv.id))
                    except StopIteration:
                        pass

            elif include_local and equation_graph.nodes[n].key and \
                    not equation_graph.nodes[n].ast_type == ast.Constant:
                var_key = equation_graph.nodes[n].key
                neq = mappings_graph.add_node(Node(key=var_key, node_type=NodeTypes.VAR,
                                                   is_set_var=is_set, ast_type=equation_graph.nodes[n].ast_type,
                                                   label=var_key),
                                              ignore_existing=True)

                end_edges = equation_graph.get_edges_for_node(end_node=n)

                try:
                    next(end_edges)
                    mappings_graph.add_edge(
                        Edge(start=eq_node_idx, end=neq, e_type=EdgeType.TARGET, arg_local=True))
                except StopIteration:
                    pass

    return eq_node_idx
