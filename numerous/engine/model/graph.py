import numpy as np
from graphviz import Digraph
from numba import njit, prange, int64, boolean, int64
from numba.experimental import jitclass
import numpy as np, networkx
from time import time
from graphviz import Digraph
from enum import IntEnum, unique
from hashlib import sha256

@njit
def multi_replace(arr, to_rep, new_val):
    for t in to_rep:
        arr[:] = np.where(arr==t, new_val, arr)


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
  #  ('parents', int64[:, :]),
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
        self.n_children_max = 100
        self.n_nodes = n_nodes
        self.nodes = np.arange(self.n_nodes)
        #print('1')
        self.node_types = node_types
        #print('2')
        self.n_edges = len(edges)
        #print('3')
        self.edges = edges
        #print('4')
        self.indegree_map = np.zeros(self.n_nodes, np.int64)
        #print('5')
        self.children = np.zeros((self.n_nodes, self.n_children_max), np.int64)
        #print('6')
        #self.parents = np.zeros((self.n_nodes, self.n_children_max), int64)
        #print('7')
        self.parent_edges = np.zeros((self.n_nodes, self.n_children_max), np.int64)
        #print('8')
        self.children_edges = np.zeros((self.n_nodes, self.n_children_max), np.int64)
        #print('9')
        self.cyclic_dependency = np.int64(-1)
        self.cyclic_path = np.zeros((0,), np.int64)
        #print('!!!')
        self.in_degree()
        #print('lll')
        self.make_children_map()
        self.topological_sorted_nodes = np.zeros(self.n_nodes, np.int64)
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
            row_pe[row_pe[0]] = i

            row_ce = self.children_edges[e[0], :]
            row_ce[0] += 1
            if row_ce[0] > self.n_children_max - 1:
                raise ValueError('More parents than allowed!')
            row_ce[row_ce[0]] = i
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
        #print('n_nodes: ', self.n_nodes)
        sorted_nodes = np.zeros(self.n_nodes, int64)*-1
        n_sorted = 0

        zero_indegree, n_zero_indegree = self.get_zero_indegree()


        while n_zero_indegree > 0:
            #print(n_zero_indegree)
            node = zero_indegree[n_zero_indegree - 1]
            #print('zin: ',node)
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

        #print('n sorted out: ', n_sorted)
        #print('n nodes: ', self.n_nodes)
        if n_sorted < self.n_nodes:
            #print('!')
            cd, self.cyclic_path = self.detect_cyclic_graph()
            #print(cd)
            self.cyclic_dependency = cd
        else:
            self.cyclic_dependency = -1

        # print(sorted_nodes)
        self.topological_sorted_nodes = sorted_nodes
        #return sorted_nodes



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

    def get_dependants_graph(self, nodes_):

        edges = np.zeros_like(self.edges)

        ix = int64(0)

        edges_visited = np.zeros(len(self.edges), dtype=int64)
        n_visited = int64(0)
        for n in nodes_:


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

class TMP:
    def __init__(self):
        self.tmp_ = 0

    def tmp(self):
        self.tmp_ += 1
        return f'tmp{self.tmp_}'

tmp = TMP().tmp

class Graph():

    def __init__(self, preallocate_items=1000):

        self.preallocate_items = preallocate_items
        self.allocated = self.preallocate_items
        self.edge_counter = 0
        self.node_counter = 0
        #Maps a key to an integer which is the internal node_id
        self.node_map = {}
        self.key_map = {}
        self.nodes_attr = {'deleted': [0]*preallocate_items}
        self.edges_attr = {'deleted': [0]*preallocate_items}
        self.edges = np.ones((self.preallocate_items, 2), dtype=np.int32)*-1
        self.lower_graph = None

        self.node_edges = None

    def build_node_edges(self):
        self.node_edges = [([],[]) for n in range(self.node_counter)]
        for i, e in enumerate(self.edges[:self.edge_counter]):
            self.node_edges[e[0]][0].append(i)
            self.node_edges[e[1]][1].append(i)


    def lock(self):
        pass

    def unlock(self):
        pass

    def add_node(self, key=None, ignore_existing=False, skip_existing=True, **attrs):
        if not key:
            key = tmp()
        if key not in self.node_map or ignore_existing:
            if not key in self.node_map:

                node = self.node_counter
                self.node_map[key] = node
                self.key_map[node] = key
                self.node_counter += 1

                if self.node_counter > self.allocated:
                    raise ValueError('Exceeding allocation')

            else:
                node = self.node_map[key]


            for ak, a in attrs.items():
                if not ak in self.nodes_attr:
                    self.nodes_attr[ak] = [None]*self.preallocate_items

                self.nodes_attr[ak][node] = a



        else:
            if not skip_existing:
                raise ValueError(f'Node with key already in graph <{key}>')
            else:
                return self.node_map[key]
        return node



    def add_edge(self, start=-1, end=-1, **attrs):
        edge = self.edge_counter
        self.edges[edge,:] = [start, end]

        self.edge_counter += 1

        if self.edge_counter > self.allocated:
            raise ValueError('Exceeding allocation')

        for ak, a in attrs.items():
            if not ak in self.edges_attr:
                self.edges_attr[ak] = [None] * self.preallocate_items

            self.edges_attr[ak][edge] = a

        return edge

    def set_edge(self, edge, start=None, end=None):
        #print(edge)
        if start:
            #print('start: ', start)
            self.edges[edge, 0] = start
        if end:
            self.edges[edge, 1] = end

    def remove_node(self, node):
        self.nodes_attr['deleted'][node] = 1

    def clean(self):
        self.lower_graph = None
        self.node_edges = None

        attr_keys = list(self.nodes_attr.keys())
        cleaned_graph = Graph(preallocate_items=self.preallocate_items)

        old_new = {n: cleaned_graph.add_node(key=k, **{a: self.nodes_attr[a][n] for a in attr_keys}) for k, n in self.node_map.items() if self.get(n, 'deleted') <=0}


        edge_keys = self.edges_attr.keys()
        for i, e in enumerate(self.edges[:self.edge_counter]):
            if e[0] in old_new and e[1] in old_new and self.edges_attr['deleted'][i]<=0:
                cleaned_graph.add_edge(old_new[e[0]], old_new[e[1]],
                          **{k: self.edges_attr[k][i] for k in edge_keys})

        return cleaned_graph

    def remove_edge(self, edge):
        self.edges_attr['deleted'][edge] = 1

    def get(self, node, attr):
        return self.nodes_attr[attr][node]

    def get_where_attr(self, attr, val, not_=False):
        if not_:
            return [i for i, v in enumerate(self.nodes_attr[attr]) if not v == val]
        else:
            if isinstance(val, list):
                return [i for i, v in enumerate(self.nodes_attr[attr]) if v in val]
            else:
                return [i for i, v in enumerate(self.nodes_attr[attr]) if v == val]

    def get_edges_for_node(self, start_node=None, end_node=None):
        if not start_node is None:
            start_ix = self.edges[:self.edge_counter,0] == start_node
            start_ = self.edges[:self.edge_counter][start_ix]
        else:
            start_ = self.edges[:self.edge_counter,:]
            start_ix = range(self.edge_counter)
        if not end_node is None:
            end_ix = start_[:, 1] == end_node
            end_ = self.edges[:self.edge_counter][end_ix]

        else:
            end_ix = start_ix
            end_ = start_

        return zip(np.argwhere(end_ix), end_)
    """
        def get_edges_for_node_filter(self, attr, start_node=None, end_node=None, val=None):
            if start_node and end_node:
                raise ValueError('arg cant have both start and end!')
            
            if not start_node is None:
                ix = np.argwhere(self.edges[:,0] == start_node)
    
    
            if not end_node is None:
                ix = np.argwhere(self.edges[:, 1] == end_node)
    
            if start_node is None and end_node is None:
                print(end_node)
                print(start_node)
                raise ValueError('Need at least one node!')
    
            ix = [i[0] for i in ix if self.edges_attr[attr][i[0]] == val]
    
            return ix, [self.edges[i,:] for i in ix]
    """


    def get_edges_for_node_filter(self, attr, start_node=None, end_node=None, val=None):
        if start_node and end_node:
            raise ValueError('arg cant have both start and end!')

        if not self.node_edges:
            self.build_node_edges()


        if not start_node is None:
            #ix = np.argwhere(self.edges[:, 0] == start_node)
            ix = self.node_edges[start_node][0]

        if not end_node is None:
            #ix = np.argwhere(self.edges[:, 1] == end_node)
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
            #return [start_node, end_node] in self.edges[:]
            #return np.where((self.edges[:self.edge_counter] == (start_node, end_node)).all(axis=1))
            return np.where((self.edges[:self.edge_counter,0]== start_node) & (self.edges[:self.edge_counter, 1] == end_node))[0]

        if start_node:
            return start_node in self.edges[:,0]

        if end_node:
            return end_node in self.edges[:, 1]




    def clone(self):
        from copy import deepcopy
        clone_ = Graph(preallocate_items=self.preallocate_items)

        clone_.preallocate_items =  self.preallocate_items
        clone_.edge_counter = self.edge_counter
        clone_.node_counter = self.node_counter
        # Maps a key to an integer which is the internal node_id
        clone_.node_map = self.node_map.copy()
        clone_.key_map = self.key_map.copy()
        clone_.nodes_attr = self.nodes_attr.copy()
        clone_.edges_attr = deepcopy(self.edges_attr)
        clone_.edges = self.edges.copy()

        return clone_

    def update(self, another_graph):

        another_keys = list(another_graph.nodes_attr.keys())
        new_map = {}
        for nk, ni in another_graph.node_map.items():
            newi = self.add_node(key=nk, **{k: another_graph.nodes_attr[k][ni] for k in another_keys}, ignore_existing=True)
            new_map[ni] = newi

        another_eqdge_keys = list(another_graph.edges_attr.keys())

        for i, e in enumerate(another_graph.edges[:another_graph.edge_counter]):

            self.add_edge(new_map[e[0]], new_map[e[1]], **{k: another_graph.edges_attr[k][i] for k in another_eqdge_keys})

    def subgraph(self, nodes, edges):
        subgraph = Graph()
        sg_map = {}
        for n in nodes:
            sg_map[n] = subgraph.add_node(self.key_map[n], **{k: v[n] for k, v in self.nodes_attr.items()})

        for e in edges:
            subgraph.add_edge(sg_map[e[0]], sg_map[e[1]])
        return subgraph

    def as_graphviz(self, file, force=False):
        #if True:
        if False or force:
            #print(self.key_map)
            #print(self.edges_attr.keys())
            dot = Digraph()

            #print('ndoses')
            for k, n in self.node_map.items():
             #   print(k)

                dot.node(k, label=self.nodes_attr['label'][n])

            for i, e in enumerate(self.edges[:self.edge_counter]):



                try:
                    if e[0]>=0 and e[1]>=0:
                        dot.edge(self.key_map[e[0]], self.key_map[e[1]], label=self.edges_attr['e_type'][i])
                except:
                    print(e)
                    raise


            dot.render(file, view=True, format='png')

    def zero_in_degree(self):

        return [n for n in self.node_map.values() if len(list(self.get_edges_for_node(end_node=n))) == 0]



    def make_lower_graph(self, top_sort=False):

        #print('!!')
        #lower_edges = self.lower_edges()
        #print('!')
        #print(self.nodes_attr['node_type'])
        #print(self.nodes_attr['node_type'][:self.node_counter])
        self.lower_graph = _Graph(self.node_counter,
                                  np.array(self.edges[:self.edge_counter], np.int64),
                                  np.array(self.nodes_attr['node_type'][:self.node_counter], np.int64))

        if top_sort:
            #print('sorting topo')
            self.lower_graph.topological_sort()

    def graph_from_path(self, path):
        cg = Graph()
        prev=None
        for p in path:
            this_ = cg.add_node(key=self.key_map[p], label=self.key_map[p], ignore_existing=True)
            if prev:
                cg.add_edge(prev, this_, e_type='dep')
            prev=this_


        return cg

    def topological_nodes(self):
        if not self.lower_graph:
            self.make_lower_graph()

        self.lower_graph.topological_sort()

        if self.lower_graph.cyclic_dependency >= 0:
            unsorted_nodes = set(self.lower_graph.nodes).difference(set(self.lower_graph.topological_sorted_nodes))
            #print('Unsorted nodes: ', unsorted_nodes)

            self.cyclic_path = self.lower_graph.cyclic_path
            #print(self.lower_graph.cyclic_path)
            cg = self.graph_from_path(self.cyclic_path)
            cg.as_graphviz('cyclic', force=True)
            for n in self.cyclic_path:
                print(" ".join([str(self.key_map[n]), '          '+str(self.get(n, 'file'))]))#, 'line: '+ str(n[1].lineno), 'col: '+str(n[1].col_offset)]))

            self.cyclic_dependency = self.lower_graph.cyclic_dependency
            raise ValueError('Cyclic path detected: ', self.cyclic_path)

        return self.lower_graph.topological_sorted_nodes

    def get_dependants_graph(self, node):
        if not self.lower_graph:
            self.make_lower_graph()

        #s_nodes = list(self.get_nodes())

        #'my nodes: ', s_nodes)
        #l_nodes = np.array([s_nodes.index(node) for node in nodes], np.int32)
        #aprint('nod: ', nodes)
        #if len(l_nodes)>0:
        nodes, edges = self.lower_graph.get_dependants_graph(np.array([node], np.int64))
        return self.subgraph(nodes, edges)
        #else:
        #    return Graph()

    def replace_nodes_by_key(self, key, to_be_replaced):
        n = self.node_map[key]
        to_be_replaced_v = np.array([self.node_map[k] for k in to_be_replaced], np.int64)
        edges = self.edges[:self.edge_counter]

        multi_replace(edges, to_be_replaced_v, n)
        [self.remove_node(t) for t in to_be_replaced_v]