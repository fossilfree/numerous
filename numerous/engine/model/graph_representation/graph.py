import logging
import numpy as np
from graphviz import Digraph
from .utils import EdgeType, TemporaryKeyGenerator
from .lower_graph import multi_replace, _Graph

tmp_generator = TemporaryKeyGenerator().generate


class Node:
    def __init__(self, key=None, ao=None, file=None, name=None, ln=None,
                 label=None, ast_type=None, node_type=None, ast_op=None):
        self.key = key
        self.ao = ao
        self.file = file
        self.name = name
        self.ln = ln
        self.label = label
        self.ast_type = ast_type
        self.node_type = node_type
        self.ast_op = ast_op
        self.deleted = False


class Graph:

    def __init__(self, preallocate_items=1000):

        self.preallocate_items = preallocate_items
        self.allocated = self.preallocate_items
        self.edge_counter = 0
        self.node_counter = 0
        # Maps a key to an integer which is the internal node_id
        self.node_map = {}
        self.key_map = {}
        self.nodes = [[Node()] * preallocate_items]
        self.nodes_attr = [[Node()] * preallocate_items]
        self.edges_attr = {'deleted': [0] * preallocate_items, "e_type": [EdgeType.UNDEFINED] * self.preallocate_items}
        self.edges = np.ones((self.preallocate_items, 2), dtype=np.int32) * -1
        self.lower_graph = None

        self.node_edges = None

        ##For equation arguments order.
        self.arg_metadata = []
        self.skipped_arg_metadata = []

    def build_node_edges(self):
        self.node_edges = [([], []) for n in range(self.node_counter)]
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
            ##TODO NODE allocation
            self.nodes[node_n] = node

        else:
            if not skip_existing:
                raise ValueError(f'Node with key already in graph <{node.key}>')
            else:
                return self.node_map[node.key]
        return node_n

    def add_edge(self, start=-1, end=-1, e_type=EdgeType.UNDEFINED, **attrs):
        edge = self.edge_counter
        self.edges[edge, :] = [start, end]

        self.edge_counter += 1

        if self.edge_counter > self.allocated:
            raise ValueError('Exceeding allocation')

        for ak, a in attrs.items():
            if not ak in self.edges_attr:
                self.edges_attr[ak] = [None] * self.preallocate_items

            self.edges_attr[ak][edge] = a
        self.edges_attr["e_type"][edge] = e_type
        if e_type == EdgeType.TARGET:
            pass
        return edge

    def set_edge(self, edge, start=None, end=None):
        # print(edge)
        if start:
            # print('start: ', start)
            self.edges[edge, 0] = start
        if end:
            self.edges[edge, 1] = end

    def remove_node(self, node_n):
        self.nodes[node_n].deleted = True

    def clean(self):
        logging.info('Cleaning eq graph')
        self.lower_graph = None
        self.node_edges = None
        ##realocate Nodes
        return self

    def remove_edge(self, edge):
        self.edges_attr['deleted'][edge] = 1

    def get(self, node, attr):
        return getattr(self.nodes[node],attr)

    def set(self, node, attr, val):
        setattr(self.nodes[node], attr, val)


    def get_where_attr(self, attr, val, not_=False):
        def filter_function(node):
            if node.deleted:
                return False
            if getattr(node, attr) == val:
                return True and not not_
            else:
                return False or not_

        return filter(filter_function, self.nodes)



    def get_edges_for_node(self, start_node=None, end_node=None):
        if not start_node is None:
            start_ix = self.edges[:self.edge_counter, 0] == start_node
            start_ = self.edges[:self.edge_counter][start_ix]
        else:
            start_ = self.edges[:self.edge_counter, :]
            start_ix = range(self.edge_counter)
        if not end_node is None:
            end_ix = start_[:, 1] == end_node
            end_ = self.edges[:self.edge_counter][end_ix]

        else:
            end_ix = start_ix
            end_ = start_

        return zip(np.argwhere(end_ix), end_)

    def get_edges_for_node_filter(self, attr, start_node=None, end_node=None, val=None):
        if start_node and end_node:
            raise ValueError('arg cant have both start and end!')

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

        if isinstance(val, list):
            ix = [i for i in ix if self.edges_attr[attr][i] in val]
        else:
            ix = [i for i in ix if self.edges_attr[attr][i] == val]

        return ix, [self.edges[i, :] for i in ix]

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
        clone_.edges_attr = deepcopy(self.edges_attr)
        clone_.edges = self.edges.copy()

        return clone_

    def update(self, another_graph):
        another_keys = list(another_graph.nodes_attr.keys())
        new_map = {}
        for nk, ni in another_graph.node_map.items():
            newi = self.add_node(key=nk, **{k: another_graph.nodes_attr[k][ni] for k in another_keys},
                                 ignore_existing=True)
            new_map[ni] = newi

        another_eqdge_keys = list(another_graph.edges_attr.keys())

        for i, e in enumerate(another_graph.edges[:another_graph.edge_counter]):
            self.add_edge(new_map[e[0]], new_map[e[1]],
                          **{k: another_graph.edges_attr[k][i] for k in another_eqdge_keys})

    def subgraph(self, nodes, edges):
        subgraph = Graph()
        sg_map = {}
        for n in nodes:
            sg_map[n] = subgraph.add_node(self.key_map[n], **{k: v[n] for k, v in self.nodes_attr.items()})

        for e in edges:
            subgraph.add_edge(sg_map[e[0]], sg_map[e[1]])
        return subgraph

    def as_graphviz(self, file, force=False):
        # if True:
        if False or force:
            dot = Digraph()
            for k, n in self.node_map.items():
                dot.node(k, label=self.nodes_attr['label'][n])

            for i, e in enumerate(self.edges[:self.edge_counter]):
                if self.edges_attr['deleted'][i] == 0:
                    try:
                        if e[0] >= 0 and e[1] >= 0:
                            dot.edge(self.key_map[e[0]], self.key_map[e[1]], label=str(self.edges_attr['e_type'][i]))
                    except:
                        print(e)
                        raise

            dot.render(file, view=True, format='pdf')

    def zero_in_degree(self):

        return [n for n in self.node_map.values() if len(list(self.get_edges_for_node(end_node=n))) == 0]

    def make_lower_graph(self, top_sort=False):
        self.lower_graph = _Graph(self.node_counter,
                                  np.array(self.edges[:self.edge_counter], np.int64),
                                  np.array(self.nodes_attr['node_type'][:self.node_counter], np.int64))

        if top_sort:
            self.lower_graph.topological_sort()

    def graph_from_path(self, path):
        cg = Graph()
        prev = None
        for p in path:
            this_ = cg.add_node(key=self.key_map[p], label=self.key_map[p], ignore_existing=True)
            if prev is not None:
                cg.add_edge(prev, this_, e_type='dep')
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
