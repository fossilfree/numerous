import logging
from copy import copy

import numpy as np
from graphviz import Digraph
from .utils import EdgeType, TemporaryKeyGenerator
from .lower_graph import multi_replace, _Graph

tmp_generator = TemporaryKeyGenerator().generate


class Node:
    def __init__(self, key=None, ao=None, file=None, name=None, ln=None,
                 label=None, ast_type=None, vectorized=None, item_id=None, node_type=None, ops=None, func=None,
                 id=None, local_id=None, scope_var=None,
                 ast_op=None, value=None, is_set_var=None, set_var_ix=None):
        self.key = key
        self.ao = ao
        self.file = file
        self.value = value
        self.name = name
        self.ln = ln
        self.label = label
        self.ast_type = ast_type
        self.node_type = node_type
        self.ast_op = ast_op
        self.id = id
        self.vectorized = vectorized
        self.item_id = item_id
        self.is_set_var = is_set_var
        self.ops = ops
        self.set_var_ix = set_var_ix
        self.func = func
        self.local_id = local_id
        self.scope_var = scope_var
        self.deleted = False
        self.node_n = -1


class Edge:
    def __init__(self, start=-1, end=-1, e_type=EdgeType.UNDEFINED, label=None, arg_local=None, mappings=None,
                 branches=None):
        self.start = start
        self.end = end
        self.e_type = e_type
        self.arg_local = arg_local
        self.branches = branches
        self.label = label
        self.mappings = mappings
        self.deleted = False
        self.edge_n = -1


class Graph:

    def __init__(self, preallocate_items=1000):

        self.preallocate_items = preallocate_items
        self.allocated = self.preallocate_items
        self.edge_counter = 0
        self.node_counter = 0
        # Maps a key to an integer which is the internal node_id
        self.node_map = {}
        self.key_map = {}
        self.nodes = []
        self.edges_c = []
        # self.edges_attr = {'deleted': [0] * preallocate_items, "e_type": [EdgeType.UNDEFINED] * self.preallocate_items}
        self.edges = np.ones((self.preallocate_items, 2), dtype=np.int32) * -1
        self.lower_graph = None

        self.node_edges = None

        ##For equation arguments order.
        self.arg_metadata = []
        self.skipped_arg_metadata = []

    def build_node_edges(self):
        self.node_edges = [([], []) for _ in range(self.node_counter)]
        for i, e in enumerate(self.edges[:self.edge_counter]):
            self.node_edges[e[0]][0].append(i)
            self.node_edges[e[1]][1].append(i)

    def add_node(self, node, ignore_existing=False, skip_existing=True):
        if not node.key:
            node.key = tmp_generator()
        if node.key not in self.node_map or ignore_existing:
            if not node.key in self.node_map:

                node_n = self.node_counter
                self.node_map[node.key] = node_n
                self.key_map[node_n] = node.key
                self.node_counter += 1

                if self.node_counter > self.allocated:
                    raise ValueError('Exceeding allocation')

            else:
                node_n = self.node_map[node.key]
            node.node_n = node_n
            if node_n < len(self.nodes):
                self.nodes[node_n] = node
            else:
                self.nodes.append(node)

        else:
            if not skip_existing:
                raise ValueError(f'Node with key already in graph <{node.key}>')
            else:
                return self.node_map[node.key]
        return node_n

    def add_edge(self, edge: Edge):
        edge_n = self.edge_counter
        self.edges[edge_n, :] = [edge.start, edge.end]

        self.edge_counter += 1

        if self.edge_counter > self.allocated:
            raise ValueError('Exceeding allocation')
        self.edges_c.append(edge)
        edge.edge_n = edge_n
        if edge.e_type == EdgeType.TARGET:
            pass
        return edge_n

    def set_edge(self, edge, start=None, end=None):
        if start:
            self.edges[edge, 0] = start
        if end:
            self.edges[edge, 1] = end

    def remove_node(self, node_n):
        self.nodes[node_n].deleted = True

    def clean(self):
        logging.info('Cleaning eq graph')
        self.lower_graph = None
        self.node_edges = None
        return self

    def remove_edge(self, edge_n):
        self.edges_c[edge_n].deleted = True

    def get(self, node, attr):
        return getattr(self.nodes[node], attr)

    def set(self, node, attr, val):
        setattr(self.nodes[node], attr, val)

    def get_where_node_attr(self, attr, val, not_=False):
        def filter_function(node):
            if node.deleted:
                return False
            if getattr(node, attr) == val:
                return True and not not_
            else:
                return False or not_

        return [node.node_n for node in filter(filter_function, self.nodes)]

    def get_edges_for_node(self, start_node=None, end_node=None):
        if start_node is not None:
            start_ix = self.edges[:self.edge_counter, 0] == start_node
            start_ = self.edges[:self.edge_counter][start_ix]
        else:
            start_ = self.edges[:self.edge_counter, :]
            start_ix = range(self.edge_counter)
        if end_node is not None:
            end_ix = start_[:, 1] == end_node
            end_ = self.edges[:self.edge_counter][end_ix]

        else:
            end_ix = start_ix
            end_ = start_

        return zip(np.argwhere(end_ix), end_)

    def get_edges_for_node_filter(self, attr, start_node=None, end_node=None, val=None):
        if start_node and end_node:
            raise ValueError('arg cant have both start and end!')
        ix = []
        if not self.node_edges:
            self.build_node_edges()

        if not start_node is None:
            ix = self.node_edges[start_node][0]

        if not end_node is None:
            ix = self.node_edges[end_node][1]

        if start_node is None and end_node is None:
            print(end_node)
            print(start_node)
            raise ValueError('Need at least one node!')

        def filter_function(edge):
            if edge.deleted:
                return False
            if getattr(edge, attr) == val or getattr(edge, attr) in val:
                return True
            else:
                return False

        ix_r = [edge.edge_n for edge in filter(filter_function, map(self.edges_c.__getitem__, ix))]
        return ix_r, [self.edges[i, :] for i in ix_r]


    def has_edge_for_nodes(self, start_node=None, end_node=None):

        if start_node and end_node:
            return np.where(
                (self.edges[:self.edge_counter, 0] == start_node) & (self.edges[:self.edge_counter, 1] == end_node))[0]

        if start_node:
            return start_node in self.edges[:, 0]

        if end_node:
            return end_node in self.edges[:, 1]

    def clone(self):
        from copy import deepcopy
        clone_ = Graph(preallocate_items=self.preallocate_items)

        clone_.preallocate_items = self.preallocate_items
        clone_.edge_counter = self.edge_counter
        clone_.node_counter = self.node_counter
        # Maps a key to an integer which is the internal node_id
        clone_.node_map = self.node_map.copy()
        clone_.key_map = self.key_map.copy()

        clone_.nodes = self.nodes.copy()
        clone_.edges_c = deepcopy(self.edges_c)
        clone_.edges = self.edges.copy()

        return clone_

    def subgraph(self, nodes, edges):
        subgraph = Graph()
        sg_map = {}
        for n in nodes:
            node_copy = copy(self.nodes[n])
            node_copy.key = self.key_map[n]
            sg_map[n] = subgraph.add_node(node_copy)

        for e in edges:
            subgraph.add_edge(Edge(start=sg_map[e[0]], end=sg_map[e[1]]))
        return subgraph

    def as_graphviz(self, file, force=False):
        if False or force:
            dot = Digraph()
            for k, n in self.node_map.items():
                dot.node(k, label=self.nodes[n].label)

            for i, e in enumerate(self.edges[:self.edge_counter]):
                if not self.edges_c[i].deleted:
                    try:
                        if e[0] >= 0 and e[1] >= 0:
                            dot.edge(self.key_map[e[0]], self.key_map[e[1]], label=str(self.edges_c[i].e_type))
                    except Exception as e:
                        print(e)
                        raise

            dot.render(file, view=True, format='pdf')

    def zero_in_degree(self):

        return [n for n in self.node_map.values() if len(list(self.get_edges_for_node(end_node=n))) == 0]

    def make_lower_graph(self, top_sort=False):
        self.lower_graph = _Graph(self.node_counter,
                                  np.array(self.edges[:self.edge_counter], np.int64),
                                  np.array([node.node_type for node in self.nodes[:self.node_counter]], np.int64))

        if top_sort:
            self.lower_graph.topological_sort()

    def graph_from_path(self, path):
        cg = Graph()
        prev = None
        for p in path:
            new_node = Node(key=self.key_map[p], label=self.key_map[p])
            this_ = cg.add_node(new_node, ignore_existing=True)
            if prev is not None:
                cg.add_edge(Edge(start=prev, end=this_, e_type=EdgeType.DEP))
            prev = this_

        return cg

    def topological_nodes(self):
        logging.info('Starting topological sort')
        if not self.lower_graph:
            self.make_lower_graph()

        self.lower_graph.topological_sort()

        if self.lower_graph.cyclic_dependency >= 0:
            unsorted_nodes = set(self.lower_graph.nodes).difference(set(self.lower_graph.topological_sorted_nodes))
            self.cyclic_path = self.lower_graph.cyclic_path
            cg = self.graph_from_path(self.cyclic_path)
            cg.as_graphviz('cyclic', force=True)
            for n in self.cyclic_path:
                print(" ".join([str(self.key_map[n]), '          ' + str(
                    self.get(n, 'file'))]))

            self.cyclic_dependency = self.lower_graph.cyclic_dependency
            raise ValueError('Cyclic path detected: ', self.cyclic_path)
        return self.lower_graph.topological_sorted_nodes

    def get_dependants_graph(self, node):
        if not self.lower_graph:
            self.make_lower_graph()
        nodes, edges = self.lower_graph.get_dependants_graph(np.array([node], np.int64))
        return self.subgraph(nodes, edges)

    def replace_nodes_by_key(self, key, to_be_replaced):
        n = self.node_map[key]
        to_be_replaced_v = np.array([self.node_map[k] for k in to_be_replaced], np.int64)
        edges = self.edges[:self.edge_counter]

        multi_replace(edges, to_be_replaced_v, n)
        [self.remove_node(t) for t in to_be_replaced_v]
