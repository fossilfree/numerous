from numba import njit, prange, intp, boolean
from numba.experimental import jitclass
import numpy as np, networkx
from time import time

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


class Graph:
    def __init__(self, nodes=None, edges=None):

        self.nodes_map = {n.id: i for i, n in enumerate(nodes)} if nodes else {}
        self.nodes = self.nodes_map.values()
        self.edges = edges if edges else []
        self.lower_graph = None

    def make_lower_graph(self):
        self.lower_graph = _Graph(len(self.nodes), self.lower_edges())

    def lower_edges(self):
        self.lowered_edges = np.zeros((len(self.edges), 2))
        for i, e in enumerate(self.edges):
            self.lowered_edges[i, :] = (self.nodes_map[e[0]], self.nodes_map[e[1]])

        return np.array(self.lowered_edges, np.int)

    def higher_nodes(self, nodes):

        return np.array(self.nodes)[nodes]

    def higher_edges(self, edges):
        return [tuple(self.nodes[e]) for e in edges]

    def get_as_lowered(self):
        lowered = self.lower_edges()
        return np.max(lowered) + 1, lowered

    def add_node(self, n, ignore_exist=False):
        if not n.id in self.nodes_map:
            self.nodes_map[n.id] = n
            self.lower_graph = None
            self.nodes = self.nodes_map.values()
        elif not ignore_exist:
            raise ValueError('Node <',n.id,'> already in graph!!')
        else:
            pass#print('node ignored')

    def add_edge(self, e, ignore_missing_nodes=False):
        #check nodes exist
        if not ignore_missing_nodes:
            if not e.start in self.nodes_map:
                raise ValueError('start node not in map! '+e.start + ' '+e.end)
            if not e.end in self.nodes_map:
                raise ValueError('end node not in map!')

        self.edges.append(e)
        self.lower_graph = None

    def topological_nodes(self):
        if not self.lower_graph:
            self.make_lower_graph()

        return self.higher_nodes(self.lower_graph.topological_sorted_nodes)

    def as_networkx_digraph(self):
        ndg = networkx.DiGraph()
        for id, n in self.nodes_map.items():
            ndg.add_node(id)
        for e in self.edges:
            if e.start and e.end:
                ndg.add_edge(e.start, e.end)

        return ndg

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
        print('update!')
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