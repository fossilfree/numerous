from numba import njit, prange, intp, boolean, int32
from numba import jitclass
import numpy as np, networkx
from time import time
from graphviz import Digraph

@njit('i4(i4[:],i4)')
def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx[0]
    return int32(-1)

@njit('Tuple((i4, i4[:]))(i4,i4[:],i4[:,:])')
def depth_first_search(node, path, children):


    children_of_node = children[node, :]
    #print('children: ', children_of_node)



    for i in range(children_of_node[0]):

        c=children_of_node[i+1]
        cycle = False
        if len(path) > 0:
            if index(path, c) >= 0:
                return c, np.append(path, c)



        path_ = np.append(path, c)


        cyclic_dependency, cyclic_path = depth_first_search(c, path_, children)
        # cyclic_dependency = -1
        if cyclic_dependency >= 0:
            return cyclic_dependency, cyclic_path

    return int32(-1), np.zeros((0,), dtype=int32)

# intp = np.int

spec = [
    ('n_children_max', intp),
    ('n_nodes', intp),
    ('n_edges', intp),
    ('nodes', intp[:]),
    ('edges', intp[:, :]),
    ('children', intp[:, :]),
    ('ancestors', intp[:, :]),
    ('indegree_map', intp[:]),
    ('topological_sorted_nodes', intp[:]),
    ('cyclic_dependency', int32),
    ('cyclic_path', int32[:])

]




@jitclass(spec)
class _Graph:
    def __init__(self, n_nodes: intp, edges):
        self.n_children_max = 20
        self.n_nodes = intp(n_nodes)
        self.nodes = np.arange(self.n_nodes)
        self.n_edges = len(edges)
        self.edges = edges
        self.indegree_map = np.zeros(self.n_nodes, intp)
        self.children = np.zeros((self.n_nodes, self.n_children_max), intp)
        self.cyclic_dependency = int32(-1)
        self.cyclic_path = np.zeros((0,), int32)
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
            if row[0] > self.n_children_max - 1:
                raise ValueError('More children than allowed!')
            row[row[0]] = e[1]
        # print(self.children)

    def get_zero_indegree(self):
        n_zero_indegree = 0
        zero_indegree = np.zeros(self.n_nodes, dtype=int32)

        for i, im in enumerate(self.indegree_map):
            if im == 0:
                zero_indegree[n_zero_indegree] = i
                n_zero_indegree += 1
        return zero_indegree, n_zero_indegree

    def topological_sort(self):
        sorted_nodes = np.zeros(self.n_nodes, intp)
        n_sorted = 0

        zero_indegree, n_zero_indegree = self.get_zero_indegree()


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
            cd, self.cyclic_path = self.detect_cyclic_graph()
            self.cyclic_dependency = cd
        else:
            self.cyclic_dependency = -1

        # print(sorted_nodes)
        self.topological_sorted_nodes = sorted_nodes


    def detect_cyclic_graph(self):
        zero_indegree, n_zero_indegree = self.get_zero_indegree()


        for i in range(n_zero_indegree):
            zid = zero_indegree[i]
            cyclic_dependency, cyclic_path = depth_first_search(zid, np.zeros((0,), dtype=int32), self.children)
            if cyclic_dependency>=0:
                print('cyclic dependency: ', cyclic_dependency)
                print('cyclic path: ', cyclic_path)
                #raise ValueError('cyclic dependency')
                return cyclic_dependency, cyclic_path

        return int32(-1), np.zeros((0,), dtype=int32)



class Graph:
    def __init__(self, nodes=None, edges=None):

        self.nodes_map = {n[0]: n for i, n in enumerate(nodes)} if nodes else {}


        #self.lower_nodes_map = {n: i for i, n in enumerate(self.get_nodes())}
        self.edges = edges if edges else []
        self.lower_graph = None

    def make_lower_graph(self):

        self.lower_graph = _Graph(len(self.get_nodes()), self.lower_edges())

    def lower_edges(self):
        self.lower_nodes_map = {n[0]: i for i, n in enumerate(self.get_nodes())}

        self.lowered_edges = np.zeros((len(self.edges), 2))
        print(self.lower_nodes_map)
        for i, e in enumerate(self.edges):
            print('e: ', e[0], e[1])
            self.lowered_edges[i, :] = (self.lower_nodes_map[e[0]], self.lower_nodes_map[e[1]],)

        return np.array(self.lowered_edges, np.int)

    def higher_nodes(self, nodes):
        #return np.array(self.nodes)[list(nodes)]
        #print(self.nodes)
        nodes_ = list(self.get_nodes())
        return [nodes_[i] for i in nodes]

    def higher_edges(self, edges):
        return [tuple(self.nodes[e]) for e in edges]

    def get_as_lowered(self):
        lowered = self.lower_edges()
        return np.max(lowered) + 1, lowered

    def get_nodes(self):
        return self.nodes_map.values()

    def add_node(self, n, ignore_exist=False):
        if not n[0] in self.nodes_map:
            self.nodes_map[n[0]] = n
            self.lower_graph = None
            self.nodes =  None
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

    def graph_from_path(self, path):
        edges = []

        prev = None
        for i, p in enumerate(path):
            if prev:
                edges.append((prev[0], p[0], '-'))
            prev = p

        return Graph([p for p in path], edges)

    def topological_nodes(self):
        import timeit

        if not self.lower_graph:
            print('lowering and sorting time: ', timeit.timeit(
            lambda: self.make_lower_graph(), number=1))
        #self.make_lower_graph()

        if self.lower_graph.cyclic_dependency >= 0:
            unsorted_nodes = self.higher_nodes(set(self.lower_graph.nodes).difference(set(self.lower_graph.topological_sorted_nodes)))
            print('Unsorted nodes: ', unsorted_nodes)

            self.cyclic_path = self.higher_nodes(self.lower_graph.cyclic_path)
            print(self.lower_graph.cyclic_path)
            cg = self.graph_from_path(self.cyclic_path)
            cg.as_graphviz('cyclic')
            for n in self.cyclic_path:
                print(" ".join([str(n[0]), '          '+str(n[1].file), 'line: '+ str(n[1].lineno), 'col: '+str(n[1].col_offset)]))

            self.cyclic_dependency = self.higher_nodes([self.lower_graph.cyclic_dependency])[0]
            raise ValueError('Cyclic path detected: ', self.cyclic_path)

        return self.higher_nodes(self.lower_graph.topological_sorted_nodes)

    def as_networkx_digraph(self):
        ndg = networkx.DiGraph()
        for id, n in self.nodes_map.items():
            ndg.add_node(id)
        for e in self.edges:
            if e.start and e.end:
                ndg.add_edge(e.start, e.end)

        return ndg

    def as_graphviz(self, file):
        dot = Digraph()
        for id, n in self.nodes_map.items():
            dot.node(n[0], label=n[0])

        print('edges: ', self.edges)
        for e in self.edges:

            if e[0] and e[1]:

                dot.edge(e[0], e[1])


        dot.render(file, view=True)



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