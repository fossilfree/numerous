from numba import njit, prange, int64, boolean, int64
from numba.experimental import jitclass
import numpy as np, networkx
from time import time
from graphviz import Digraph
from enum import IntEnum, unique
from hashlib import sha256


@njit('i8(i8[:],i8)', cache=False)
def index(array, item):
    for ix, val in np.ndenumerate(array):
        if val == item:
            return int64(ix[0])
    return int64(-1)

@njit('Tuple((i8, i8[:]))(i8,i8[:],i8[:,:])', cache=False)
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

    return int64(-1), np.zeros((0,), dtype=int64)

@njit('Tuple((i8, i8, i8))(i8[:,:],i8[:,:],i8,i8[:,:],i8,i8[:],i8,i8[:], i8, i8[:])', cache=False)
def walk_parents(parent_edges, self_edges, n, edges, ix, visited_edges, n_visited, node_types, deriv_dep_count, deriv_dep):
    for i in range(parent_edges[n,0]):

        e = self_edges[parent_edges[n, i+1]]

        if index(visited_edges[:n_visited], e[2]) <0:



            visited_edges[n_visited] = e[2]
            n_visited += 1

            edges[ix,:]=e
            ix+=1
            #print(node_types[n])
            if node_types[e[0]] < 3:

                ix, n_visited, deriv_dep_count = walk_parents(parent_edges, self_edges, e[0], edges, ix, visited_edges, n_visited, node_types, deriv_dep_count, deriv_dep)
            elif node_types[e[0]] == 3:
                deriv_dep[deriv_dep_count] = e[0]
                deriv_dep_count += 1


    return ix, n_visited, deriv_dep_count

@njit('Tuple((i8, i8))(i8[:,:],i8[:,:],i8,i8[:,:],i8,i8[:],i8,i8[:])', cache=False)
def walk_parents_to_var(parent_edges, self_edges, n, edges, ix, visited_edges, n_visited, node_types):
    for i in range(parent_edges[n, 0]):
        e = self_edges[parent_edges[n, i+1]]
        if index(visited_edges[:n_visited], e[2]) < 0:


            #if e[1] == n:
            visited_edges[n_visited] = e[2]
            n_visited += 1

            edges[ix, :] = e
            ix += 1



            if node_types[e[0]]<2:
                ix, n_visited = walk_parents_to_var(parent_edges, self_edges, e[0], edges, ix, visited_edges, n_visited, node_types)


    return ix, n_visited

@njit('Tuple((i8, i8))(i8[:,:],i8[:,:],i8[:,:],i8,i8[:,:],i8,i8[:],i8,i8[:])', cache=False)
def walk_children(parent_edges, children_edges, self_edges, n, edges, ix, visited_edges, n_visited, node_types):
    for i in range(children_edges[n, 0]):

        e = self_edges[children_edges[n, i+1]]

        if index(visited_edges[:n_visited], e[2]) <0:


            #if e[0] == n:

                visited_edges[n_visited] = e[2]
                n_visited += 1

                edges[ix,:]=e
                ix+=1
                ix, n_visited = walk_children(parent_edges, children_edges, self_edges, e[1], edges, ix, visited_edges, n_visited, node_types)

            #if e[1] == n:
                if node_types[e[0]]<2:
                    ix, n_visited = walk_parents_to_var(parent_edges, self_edges, e[0], edges, ix, visited_edges, n_visited, node_types)
                #visited_edges[n_visited] = i
                #n_visited += 1

                #edges[ix,:]=e
                #ix+=1



    return ix, n_visited

@njit('Tuple((i8, i8))(i8[:,:],i8,i8[:,:],i8,i8[:],i8,i8[:])', cache=False)
def walk_parents_to_var_(self_edges, n, edges, ix, visited_edges, n_visited, node_types):
    for e in self_edges:
        if index(visited_edges[:n_visited], e[2]) < 0:


            if e[1] == n:
                visited_edges[n_visited] = e[2]
                n_visited += 1

                edges[ix, :] = e
                ix += 1



                if node_types[e[0]]<2:
                    ix, n_visited = walk_parents_to_var_(self_edges, e[0], edges, ix, visited_edges, n_visited, node_types)


    return ix, n_visited

@njit('Tuple((i8, i8))(i8[:,:],i8,i8[:,:],i8,i8[:],i8,i8[:])', cache=False)
def walk_children_(self_edges, n, edges, ix, visited_edges, n_visited, node_types):
    for e in self_edges:

        if index(visited_edges[:n_visited], e[2]) <0:


            if e[0] == n:

                visited_edges[n_visited] = e[2]
                n_visited += 1

                edges[ix,:]=e
                ix+=1
                ix, n_visited = walk_children_(self_edges, e[1], edges, ix, visited_edges, n_visited, node_types)


                if node_types[e[0]]<2:
                    ix, n_visited = walk_parents_to_var_(self_edges, e[0], edges, ix, visited_edges, n_visited, node_types)

    return ix, n_visited
# int64 = np.int

spec = [
    ('n_children_max', int64),
    ('n_nodes', int64),
    ('n_edges', int64),
    ('nodes', int64[:]),
    ('node_types', int64[:]),
    ('edges', int64[:, :]),
    ('children', int64[:, :]),
    ('parents', int64[:, :]),
    ('parent_edges', int64[:, :]),
    ('children_edges', int64[:, :]),
    ('ancestors', int64[:, :]),
    ('indegree_map', int64[:]),
    ('topological_sorted_nodes', int64[:]),
    ('cyclic_dependency', int64),
    ('cyclic_path', int64[:])

]


@jitclass(spec)
class _Graph:
    def __init__(self, n_nodes: int64, edges: int64[:], node_types):
        #print('!!!!')
        self.n_children_max = 30
        self.n_nodes = n_nodes
        self.nodes = np.arange(self.n_nodes)
        #print('1')
        self.node_types = node_types
        #print('2')
        self.n_edges = len(edges)
        #print('3')
        self.edges = edges
        #print('4')
        self.indegree_map = np.zeros(self.n_nodes, int64)
        #print('5')
        self.children = np.zeros((self.n_nodes, self.n_children_max), int64)
        #print('6')
        #self.parents = np.zeros((self.n_nodes, self.n_children_max), int64)
        #print('7')
        self.parent_edges = np.zeros((self.n_nodes, self.n_children_max), int64)
        #print('8')
        self.children_edges = np.zeros((self.n_nodes, self.n_children_max), int64)
        #print('9')
        self.cyclic_dependency = int64(-1)
        self.cyclic_path = np.zeros((0,), int64)
        #print('!!!')
        self.in_degree()
        #print('lll')
        self.make_children_map()
        #print('herer')


    def in_degree(self):

        for j in range(self.n_edges):
            self.indegree_map[self.edges[j][1]] += 1
        return self.indegree_map

    def make_children_map(self):

        for i, e in enumerate(self.edges):
            row = self.children[e[0], :]
            row[0] += 1
            if row[0] > self.n_children_max - 1:
                raise ValueError('More children than allowed!')
            row[row[0]] = e[1]

            #row_p = self.parents[e[1], :]
            #row_p[0] += 1
            #if row_p[0] > self.n_children_max - 1:
            #    raise ValueError('More parents than allowed!')
            #row_p[row_p[0]] = e[0]

            row_pe = self.parent_edges[e[1], :]
            row_pe[0] += 1
            if row_pe[0] > self.n_children_max - 1:
                raise ValueError('More parents than allowed!')
            row_pe[row_pe[0]] = e[2]

            row_ce = self.children_edges[e[0], :]
            row_ce[0] += 1
            if row_ce[0] > self.n_children_max - 1:
                raise ValueError('More parents than allowed!')
            row_ce[row_ce[0]] = e[2]
    """
    def make_edges_map(self, edges, nodes):
        n_nodes = len(nodes)
        parent_edges = np.zeros((n_nodes, self.n_children_max), int64)
        children_edges = np.zeros((n_nodes, self.n_children_max), int64)

        for i, e in enumerate(edges):


            row_pe = parent_edges[e[1], :]
            row_pe[0] += 1
            if row_pe[0] > self.n_children_max - 1:
                raise ValueError('More parents than allowed!')
            row_pe[row_pe[0]] = e[2]

            row_ce = children_edges[e[0], :]
            row_ce[0] += 1
            if row_ce[0] > self.n_children_max - 1:
                raise ValueError('More parents than allowed!')
            row_ce[row_ce[0]] = e[2]

        return parent_edges, children_edges
    """

    def get_zero_indegree(self):
        n_zero_indegree = 0
        zero_indegree = np.zeros(self.n_nodes, dtype=int64)

        for i, im in enumerate(self.indegree_map):
            if im == 0:
                zero_indegree[n_zero_indegree] = i
                n_zero_indegree += 1
        return zero_indegree, n_zero_indegree

    def topological_sort(self):

        sorted_nodes = np.zeros(self.n_nodes, int64)
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




    def get_ancestor_graph(self, n):
        edges = np.zeros_like(self.edges)
        edges_visited = np.zeros(len(self.edges), dtype=int64)
        n_visited = int64(0)
        ix = int64(0)
        dep_derivatives =  np.zeros(len(self.edges), dtype=int64)
        ix, n_visited, deriv_dep = walk_parents(self.parent_edges, self.edges, n, edges, ix, edges_visited, n_visited, self.node_types, 0, dep_derivatives)
        edges = edges[:ix,:]
        edges[:ix, 3] = 1


        nodes = np.zeros(2*len(edges), dtype=int64)

        for i, e in enumerate(edges):
            nodes[2*i] = e[0]
            nodes[2*i+1] = e[1]

        nodes = nodes[:i * 2+1+1]

        nodes = np.unique(nodes)


        return nodes, edges, dep_derivatives[:deriv_dep]

    def get_dependants_graph(self, nodes):

        edges = np.zeros_like(self.edges)

        ix = int64(0)

        edges_visited = np.zeros(len(self.edges), dtype=int64)
        n_visited = int64(0)
        for n in nodes:


            ix, n_visited = walk_children(self.parent_edges, self.children_edges, self.edges, n, edges, ix, edges_visited, n_visited, self.node_types)

        edges = edges[:ix,:]
        edges[:ix, 3] = 2

        #nodes = np.zeros((0,1),int64)
        nodes = np.zeros(2 * len(edges), dtype=int64)

        for i, e in enumerate(edges):
            nodes[2 * i] = e[0]
            nodes[2 * i + 1] = e[1]

        nodes = nodes[:i * 2+1+1]

        nodes = np.unique(nodes)

        return nodes, edges

    def get_dependants_graph_subgraph(self, nodes, subedges):

        #parent_edges, children_edges = self.make_edges_map(subedges, nodes)

        edges = np.zeros_like(subedges)

        ix = int64(0)

        edges_visited = np.zeros(len(subedges), dtype=int64)
        n_visited = int64(0)
        for n in nodes:

            ix, n_visited = walk_children_(subedges, n, edges, ix, edges_visited, n_visited, self.node_types)

        edges = edges[:ix,:]
        edges[:ix, 3] = 2

        #nodes = np.zeros((0,1),int64)
        nodes = np.zeros(2 * len(edges), dtype=int64)

        for i, e in enumerate(edges):
            nodes[2 * i] = e[0]
            nodes[2 * i + 1] = e[1]

        #nodes = nodes[:i * 2+1+1]

        nodes = np.unique(nodes)



        return nodes, edges

    def get_anc_dep_graph(self, n):
        anc_nodes, anc_edges, deriv_dependencies = self.get_ancestor_graph(n)
        #print(anc_nodes)
        state_nodes = np.array([ancn for ancn in anc_nodes if self.node_types[ancn]>=3], np.int64)
        #print('states: ', state_nodes)
        if len(state_nodes)>0:
            nodes, edges = self.get_dependants_graph_subgraph(state_nodes, anc_edges)

        else:
            nodes = anc_nodes
            edges = anc_edges

        return nodes, edges, anc_nodes, anc_edges, deriv_dependencies

    def detect_cyclic_graph(self):
        zero_indegree, n_zero_indegree = self.get_zero_indegree()


        for i in range(n_zero_indegree):
            zid = zero_indegree[i]
            cyclic_dependency, cyclic_path = depth_first_search(zid, np.zeros((0,), dtype=int64), self.children)
            if cyclic_dependency>=0:
                print('cyclic dependency: ', cyclic_dependency)
                print('cyclic path: ', cyclic_path)
                #raise ValueError('cyclic dependency')
                return cyclic_dependency, cyclic_path

        return int64(-1), np.zeros((0,), dtype=int64)

from numerous.engine.variables import VariableType


class Graph:
    def __init__(self, nodes=[], edges=None):

        if len(nodes) >0:
            #print('ss', nodes)
            self.nodes_map = {n[0]: n for i, n in enumerate(nodes)}
        else:
            self.nodes_map = {}


        #self.lower_nodes_map = {n: i for i, n in enumerate(self.get_nodes())}
        if edges:
            for e in edges:
                if len(e)<3:
                    ValueError('Missing stuff: ', str(e))

        self.edges = edges if edges else []

        self.lower_graph = None

    def make_lower_graph(self, top_sort=False):
        nodes = self.get_nodes()
        node_types = np.array([0 if isinstance(n, str) else n[1].node_type for n in nodes],np.int64)
        #print('!!')
        lower_edges = self.lower_edges()
        #print('!')
        self.lower_graph = _Graph(len(nodes), lower_edges, node_types)

        if top_sort:
            #print('sorting topo')
            self.lower_graph.topological_sort()

    def in_degree(self):
        nodes = self.higher_nodes(self.lower_graph.nodes)
        in_degree_map = {}
        return {n[0]: in_deg for n, in_deg in zip(nodes, self.lower_graph.in_degree())}



    def lower_edges(self):

        self.lower_nodes_map = {n if isinstance(n, str) else n[0]: i for i, n in enumerate(self.get_nodes())}

        if len(self.edges)>0:
            self.lowered_edges= [(self.lower_nodes_map[e[0]], self.lower_nodes_map[e[1]], i, 0) for i, e in enumerate(self.edges)]
        else:
            self.lowered_edges = np.zeros((0,4), np.int64)

        return np.array(self.lowered_edges, np.int64)

    def higher_nodes(self, nodes):
        #return np.array(self.nodes)[list(nodes)]
        #print(self.nodes)
        nodes_ = list(self.get_nodes())
        return [nodes_[i] for i in nodes]

    def higher_edges(self, edges, update_whereused=False):
        #nodes_ = list(self.get_nodes())
        if update_whereused:
            for e in edges:

                e_ = self.edges[e[2]]
                self.edges[e[2]] = (e_[0], e_[1], e_[2], e[3])

        return [self.edges[e[2]] for e in edges]

    def get_as_lowered(self):
        lowered = self.lower_edges()
        return np.max(lowered) + 1, lowered

    def get_nodes(self):
        #print('get nodes: ', list(self.nodes_map.values()))
        return list(self.nodes_map.values())

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

        if len(e)<3:
            raise ValueError('missing something ', str(e))

        self.edges.append(e)
        self.lower_graph = None

    def edges_end(self, node, label=None):
        found = []

        for e in self.edges:
            if e[1] == node[0]:
                #print('edge: ',e)
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

        #if not self.lower_graph:
        #    print('lowering and sorting time: ', timeit.timeit(
         #   lambda: self.make_lower_graph(top_sort=True), number=1))
        self.make_lower_graph(top_sort=True)

        if self.lower_graph.cyclic_dependency >= 0:
            unsorted_nodes = self.higher_nodes(set(self.lower_graph.nodes).difference(set(self.lower_graph.topological_sorted_nodes)))
            #print('Unsorted nodes: ', unsorted_nodes)

            self.cyclic_path = self.higher_nodes(self.lower_graph.cyclic_path)
            #print(self.lower_graph.cyclic_path)
            cg = self.graph_from_path(self.cyclic_path)
            cg.as_graphviz('cyclic')
            for n in self.cyclic_path:
                print(" ".join([str(n[0]), '          '+str(n[1].file), 'line: '+ str(n[1].lineno), 'col: '+str(n[1].col_offset)]))

            self.cyclic_dependency = self.higher_nodes([self.lower_graph.cyclic_dependency])[0]
            raise ValueError('Cyclic path detected: ', self.cyclic_path)
        return self.higher_nodes(self.lower_graph.topological_sorted_nodes)

    def get_ancestor_dependents_graph(self, node):


        if not self.lower_graph:
            self.make_lower_graph()
        nodes, edges, anc_nodes, anc_edges, deriv_dependencies = self.lower_graph.get_anc_dep_graph(list(self.get_nodes()).index(node))
        print('dep deriv: ', deriv_dependencies)
        nodes = self.higher_nodes(nodes)
        #TODO this is inefficient!
        edges_ = self.higher_edges(anc_edges, update_whereused=True)
        edges = self.higher_edges(edges, update_whereused=True)
        return Graph(self.higher_nodes(anc_nodes), edges_), Graph(nodes, edges), self.higher_nodes(deriv_dependencies)


    def get_ancestor_graph(self, node):
        #print('making lower graph')
        if not self.lower_graph:
            self.make_lower_graph()
        nodes, edges = self.lower_graph.get_ancestor_graph(list(self.get_nodes()).index(node))
        nodes = self.higher_nodes(nodes)
        edges = self.higher_edges(edges, update_whereused=True)
        return Graph(nodes, edges)

    def get_dependants_graph(self, nodes):
        if not self.lower_graph:
            self.make_lower_graph()

        s_nodes = list(self.get_nodes())

        #'my nodes: ', s_nodes)
        l_nodes = [s_nodes.index(node) for node in nodes]
        #aprint('nod: ', nodes)
        if len(l_nodes):
            nodes, edges = self.lower_graph.get_dependants_graph(l_nodes)
            nodes = self.higher_nodes(nodes)
            edges = self.higher_edges(edges, update_whereused=True)
            return Graph(nodes, edges)
        else:
            return Graph()

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

    def hash(self):
        if not self.lower_graph:
            self.make_lower_graph()

        e_list = []
        for e in self.lower_graph.edges:
            e_list.append(e[0])
            e_list.append(e[1])

        hash_ = hash(tuple(list(self.lower_graph.nodes) + list(self.lower_graph.node_types)+e_list))

        return hash_



"""
@njit
def in_degree_(n_nodes, edges):
    n_edges = len(edges)
    nodes = np.zeros(n_nodes,int64)



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
    children = np.zeros((n_nodes,10), int64)

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
    indegree_map = np.zeros(n_nodes, int64)
    sorted_nodes = np.zeros(n_nodes, int64)
    n_sorted = 0

    children = children_(edges, n_nodes)


    for j in range(n_edges):
        indegree_map[edges[j][1]] += 1

    n_zero_indegree = 0
    zero_indegree = np.zeros(n_nodes, int64)
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