from numba import njit, prange, intp, boolean
from numba import jitclass
import numpy as np, networkx
from time import time
from graphviz import Digraph



# intp = np.int

spec = [
    ('n_nodes', intp),
    ('n_edges', intp),
    ('nodes', intp[:]),
    ('edges', intp[:, :]),
    ('children', intp[:, :]),
    ('ancestors', intp[:, :]),
    ('indegree_map', intp[:]),
    ('topological_sorted_nodes', intp[:]),
]


@jitclass(spec)
class _Graph:
    def __init__(self, n_nodes: intp, edges):
        self.n_nodes = intp(n_nodes)
        self.nodes = np.arange(self.n_nodes)
        self.n_edges = len(edges)
        self.edges = edges
        self.indegree_map = np.zeros(self.n_nodes, intp)
        self.children = np.zeros((self.n_nodes, 10), intp)

        self.in_degree()
        self.make_children_map()
        self.topological_sort()

    def in_degree(self):
        for j in range(self.n_edges):
            self.indegree_map[self.edges[j][1]] += 1
        # print('in: ',self.indegree_map)

    def make_children_map(self):

        for e in self.edges:
            row = self.children[e[0], :]
            row[0] += 1
            if row[0] > 10 - 1:
                raise ValueError('arg')
            row[row[0]] = e[1]
        # print(self.children)

    def topological_sort(self):
        sorted_nodes = np.zeros(self.n_nodes, intp)
        n_sorted = 0

        n_zero_indegree = 0
        zero_indegree = np.zeros(self.n_nodes, intp)

        for i, im in enumerate(self.indegree_map):
            if im == 0:
                zero_indegree[n_zero_indegree] = i
                n_zero_indegree += 1

        while n_zero_indegree > 0:
            node = zero_indegree[n_zero_indegree - 1]
            sorted_nodes[n_sorted] = node
            n_sorted += 1
            children_of_node = self.children[node, :]
            n_zero_indegree -= 1
            for i in range(children_of_node[0]):
                child = children_of_node[i + 1]
                self.indegree_map[child] -= 1
                if self.indegree_map[child] == 0:
                    zero_indegree[n_zero_indegree] = child
                    n_zero_indegree += 1

        if n_sorted < self.n_nodes:
            raise ValueError('Non-feasible network')
        # print(sorted_nodes)
        self.topological_sorted_nodes = sorted_nodes

    def


class Graph:
    def __init__(self, nodes=None, edges=None):

        self.nodes_map = {n.id: i for i, n in enumerate(nodes)} if nodes else {}

        self.nodes = list(self.nodes_map.values())
        self.lower_nodes_map = {n: i for i, n in enumerate(self.nodes)}
        self.edges = edges if edges else []
        self.lower_graph = None

    def make_lower_graph(self):

        self.lower_graph = _Graph(len(self.nodes), self.lower_edges())

    def lower_edges(self):
        self.lower_nodes_map = {n[0]: i for i, n in enumerate(self.nodes)}

        self.lowered_edges = np.zeros((len(self.edges), 2))
        for i, e in enumerate(self.edges):
            self.lowered_edges[i, :] = (self.lower_nodes_map[e[0]], self.lower_nodes_map[e[1]],)

        return np.array(self.lowered_edges, np.int)

    def higher_nodes(self, nodes):
        #return np.array(self.nodes)[list(nodes)]
        #print(self.nodes)
        return [self.nodes[i] for i in nodes]

    def higher_edges(self, edges):
        return [tuple(self.nodes[e]) for e in edges]

    def get_as_lowered(self):
        lowered = self.lower_edges()
        return np.max(lowered) + 1, lowered

    def add_node(self, n, ignore_exist=False):
        if not n[0] in self.nodes_map:
            self.nodes_map[n[0]] = n
            self.lower_graph = None
            self.nodes =  list(self.nodes_map.values())
        elif not ignore_exist:
            raise ValueError('Node <',n[0],'> already in graph!!')
        else:
            pass#print('node ignored')

    def add_edge(self, e, ignore_missing_nodes=False):
        #check nodes exist
        if not ignore_missing_nodes:
            if not e[0] in self.nodes_map:
                raise ValueError('start node not in map! '+e[0] + ' '+e[1])
            if not e[1] in self.nodes_map:
                raise ValueError('end node not in map!')

        self.edges.append(e)
        self.lower_graph = None

    def edges_end(self, node, label=None):
        found = []

        for e in self.edges:
            if e[1] == node[0]:
                if not label or label in e[2].label:
                    found.append(e)
        return found

    def edges_start(self, node, label=None):
        found = []

        for e in self.edges:
            if e[0] == node[0]:
                if not label or label in e[2].label:
                    found.append(e)
        return found


    def topological_nodes(self):
        import timeit

        if not self.lower_graph:
            print('lowering and sorting time: ', timeit.timeit(
            lambda: self.make_lower_graph(), number=1))
        #self.make_lower_graph()

        return self.higher_nodes(self.lower_graph.topological_sorted_nodes)

    def as_networkx_digraph(self):
        ndg = networkx.DiGraph()
        for id, n in self.nodes_map.items():
            ndg.add_node(id)
        for e in self.edges:
            if e.start and e.end:
                ndg.add_edge(e.start, e.end)

        return ndg

    def as_graphviz(self):
        dot = Digraph()
        for id, n in self.nodes_map.items():
            dot.node(id, label=n[1].label)
        for e in self.edges:
            if e[0] and e[1]:

                dot.edge(e[0], e[1])


        dot.render('g.gv', view=True)



    def draw_graph(self):



        ndg = self.as_networkx_digraph()
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        networkx.draw_networkx(ndg, with_labels=True)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
        for pos in ['right', 'top', 'bottom', 'left']:
            plt.gca().spines[pos].set_visible(False)

        plt.show()

    def set_node_map(self, node_map):
        self.nodes_map = node_map
        self.nodes = self.nodes_map.values()

    def update(self, other):
        #print('update!')
        self.nodes_map.update(other.nodes_map)
        self.nodes = self.nodes_map.values()
        for e in other.edges:
            self.add_edge(e)

"""
@njit
def in_degree_(n_nodes, edges):
    n_edges = len(edges)
    nodes = np.zeros(n_nodes,intp)



    for j in range(n_edges):
        nodes[edges[j][1]] += 1

    return nodes

def in_degree(g: Graph):
    # Loop over all nodes and check if they are children in the edges
    n, edges = g.get_as_lowered()
    nodes_w_in_degree = in_degree_(n,edges)

    return nodes_w_in_degree, edges

@njit
def children_(edges, n_nodes):
    children = np.zeros((n_nodes,10), intp)

    for e in edges:
        row = children[e[0],:]
        row[0]+=1
        if row[0] > 10-1:
            raise ValueError('arg')
        row[row[0]] = e[1]
    return children

@njit
def successors(children, node):
    return children[node][1:children[node][0]+1]

@njit
def topological_sort_(n_nodes, edges):
    n_edges = len(edges)
    indegree_map = np.zeros(n_nodes, intp)
    sorted_nodes = np.zeros(n_nodes, intp)
    n_sorted = 0

    children = children_(edges, n_nodes)


    for j in range(n_edges):
        indegree_map[edges[j][1]] += 1

    n_zero_indegree = 0
    zero_indegree = np.zeros(n_nodes, intp)
    for i, im in enumerate(indegree_map):
        if im == 0:
            zero_indegree[n_zero_indegree] = i
            n_zero_indegree +=1



    while n_zero_indegree>0:
        node = zero_indegree[n_zero_indegree-1]
        sorted_nodes[n_sorted]=node
        n_sorted+=1
        children_of_node = children[node,:]
        n_zero_indegree-=1
        for i in range(children_of_node[0]):
            child = children_of_node[i+1]
            indegree_map[child] -= 1
            if indegree_map[child] == 0:
                zero_indegree[n_zero_indegree] = child
                n_zero_indegree += 1

    if n_sorted < n_nodes:
        raise ValueError('Non-feasible network')
    return sorted_nodes

def topological_sort(g: Graph):
    tic = time()
    n, edges = g.get_as_lowered()
    toc = time()
    print('numerous lowering time: ', toc - tic)
    tic = time()
    ts=topological_sort_(n, edges)
    toc = time()
    print('numerous sorting time: ', toc - tic)
    return ts

"""