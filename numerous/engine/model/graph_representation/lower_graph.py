from numba import njit,int64
from numba.experimental import jitclass
import numpy as np

@njit
def multi_replace(arr, to_rep, new_val):
    for t in to_rep:
        arr[:] = np.where(arr == t, new_val, arr)


@njit('i8(i8[:],i8)', cache=False)
def index(array, item):
    for ix, val in np.ndenumerate(array):
        if val == item:
            return int64(ix[0])
    return int64(-1)


@njit('Tuple((i8, i8[:]))(i8,i8[:],i8[:,:])', cache=False)
def depth_first_search(node, path, children):
    children_of_node = children[node, :]
    for i in range(children_of_node[0]):

        c = children_of_node[i + 1]
        if len(path) > 0:
            if index(path, c) >= 0:
                return c, np.append(path, c)

        path_ = np.append(path, c)

        cyclic_dependency, cyclic_path = depth_first_search(c, path_, children)
        if cyclic_dependency >= 0:
            return cyclic_dependency, cyclic_path

    return int64(-1), np.zeros((0,), dtype=int64)


@njit('Tuple((i8, i8, i8))(i8[:,:],i8[:,:],i8,i8[:,:],i8,i8[:],i8,i8[:], i8, i8[:])', cache=False)
def walk_parents(parent_edges, self_edges, n, edges, ix, visited_edges, n_visited, node_types, deriv_dep_count,
                 deriv_dep):
    for i in range(parent_edges[n, 0]):

        e = self_edges[parent_edges[n, i + 1]]

        if index(visited_edges[:n_visited], e[2]) < 0:

            visited_edges[n_visited] = e[2]
            n_visited += 1

            edges[ix, :] = e
            ix += 1
            if node_types[e[0]] < 3:

                ix, n_visited, deriv_dep_count = walk_parents(parent_edges, self_edges, e[0], edges, ix, visited_edges,
                                                              n_visited, node_types, deriv_dep_count, deriv_dep)
            elif node_types[e[0]] == 3:
                deriv_dep[deriv_dep_count] = e[0]
                deriv_dep_count += 1

    return ix, n_visited, deriv_dep_count


@njit('Tuple((i8, i8))(i8[:,:],i8[:,:],i8,i8[:,:],i8,i8[:],i8,i8[:])', cache=False)
def walk_parents_to_var(parent_edges, self_edges, n, edges, ix, visited_edges, n_visited, node_types):
    for i in range(parent_edges[n, 0]):
        e = self_edges[parent_edges[n, i + 1]]
        if index(visited_edges[:n_visited], e[2]) < 0:

            # if e[1] == n:
            visited_edges[n_visited] = e[2]
            n_visited += 1

            edges[ix, :] = e
            ix += 1

            if node_types[e[0]] < 2:
                ix, n_visited = walk_parents_to_var(parent_edges, self_edges, e[0], edges, ix, visited_edges, n_visited,
                                                    node_types)

    return ix, n_visited


@njit('Tuple((i8, i8))(i8[:,:],i8[:,:],i8[:,:],i8,i8[:,:],i8,i8[:],i8,i8[:])', cache=False)
def walk_children(parent_edges, children_edges, self_edges, n, edges, ix, visited_edges, n_visited, node_types):
    for i in range(children_edges[n, 0]):

        e = self_edges[children_edges[n, i + 1]]

        if index(visited_edges[:n_visited], e[2]) < 0:

            # if e[0] == n:

            visited_edges[n_visited] = e[2]
            n_visited += 1

            edges[ix, :] = e
            ix += 1
            ix, n_visited = walk_children(parent_edges, children_edges, self_edges, e[1], edges, ix, visited_edges,
                                          n_visited, node_types)

            # if e[1] == n:
            if node_types[e[0]] < 2:
                ix, n_visited = walk_parents_to_var(parent_edges, self_edges, e[0], edges, ix, visited_edges, n_visited,
                                                    node_types)


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

                if node_types[e[0]] < 2:
                    ix, n_visited = walk_parents_to_var_(self_edges, e[0], edges, ix, visited_edges, n_visited,
                                                         node_types)

    return ix, n_visited


@njit('Tuple((i8, i8))(i8[:,:],i8,i8[:,:],i8,i8[:],i8,i8[:])', cache=False)
def walk_children_(self_edges, n, edges, ix, visited_edges, n_visited, node_types):
    for e in self_edges:

        if index(visited_edges[:n_visited], e[2]) < 0:

            if e[0] == n:

                visited_edges[n_visited] = e[2]
                n_visited += 1

                edges[ix, :] = e
                ix += 1
                ix, n_visited = walk_children_(self_edges, e[1], edges, ix, visited_edges, n_visited, node_types)

                if node_types[e[0]] < 2:
                    ix, n_visited = walk_parents_to_var_(self_edges, e[0], edges, ix, visited_edges, n_visited,
                                                         node_types)

    return ix, n_visited


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

        self.n_children_max = 100
        self.n_nodes = n_nodes
        self.nodes = np.arange(self.n_nodes)

        self.node_types = node_types

        self.n_edges = len(edges)

        self.edges = edges

        self.indegree_map = np.zeros(self.n_nodes, np.int64)

        self.children = np.zeros((self.n_nodes, self.n_children_max), np.int64)

        # self.parents = np.zeros((self.n_nodes, self.n_children_max), int64)

        self.parent_edges = np.zeros((self.n_nodes, self.n_children_max), np.int64)

        self.children_edges = np.zeros((self.n_nodes, self.n_children_max), np.int64)

        self.cyclic_dependency = np.int64(-1)
        self.cyclic_path = np.zeros((0,), np.int64)

        self.in_degree()

        self.make_children_map()
        self.topological_sorted_nodes = np.zeros(self.n_nodes, np.int64)


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

            # row_p = self.parents[e[1], :]
            # row_p[0] += 1
            # if row_p[0] > self.n_children_max - 1:
            #    raise ValueError('More parents than allowed!')
            # row_p[row_p[0]] = e[0]

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
        # print('n_nodes: ', self.n_nodes)
        sorted_nodes = np.zeros(self.n_nodes, int64) * -1
        n_sorted = 0

        zero_indegree, n_zero_indegree = self.get_zero_indegree()

        while n_zero_indegree > 0:
            # print(n_zero_indegree)
            node = zero_indegree[n_zero_indegree - 1]
            # print('zin: ',node)
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

        # print('n sorted out: ', n_sorted)
        # print('n nodes: ', self.n_nodes)
        if n_sorted < self.n_nodes:
            # print('!')
            cd, self.cyclic_path = self.detect_cyclic_graph()
            # print(cd)
            self.cyclic_dependency = cd
        else:
            self.cyclic_dependency = -1

        # print(sorted_nodes)
        self.topological_sorted_nodes = sorted_nodes
        # return sorted_nodes

    def get_ancestor_graph(self, n):
        edges = np.zeros_like(self.edges)
        edges_visited = np.zeros(len(self.edges), dtype=int64)
        n_visited = int64(0)
        ix = int64(0)
        dep_derivatives = np.zeros(len(self.edges), dtype=int64)
        ix, n_visited, deriv_dep = walk_parents(self.parent_edges, self.edges, n, edges, ix, edges_visited, n_visited,
                                                self.node_types, 0, dep_derivatives)
        edges = edges[:ix, :]
        edges[:ix, 3] = 1

        nodes = np.zeros(2 * len(edges), dtype=int64)

        for i, e in enumerate(edges):
            nodes[2 * i] = e[0]
            nodes[2 * i + 1] = e[1]

        nodes = nodes[:i * 2 + 1 + 1]

        nodes = np.unique(nodes)

        return nodes, edges, dep_derivatives[:deriv_dep]

    def get_dependants_graph(self, nodes_):

        edges = np.zeros_like(self.edges)

        ix = int64(0)

        edges_visited = np.zeros(len(self.edges), dtype=int64)
        n_visited = int64(0)
        for n in nodes_:
            ix, n_visited = walk_children(self.parent_edges, self.children_edges, self.edges, n, edges, ix,
                                          edges_visited, n_visited, self.node_types)

        edges = edges[:ix, :]
        edges[:ix, 3] = 2

        # nodes = np.zeros((0,1),int64)
        nodes = np.zeros(2 * len(edges), dtype=int64)

        for i, e in enumerate(edges):
            nodes[2 * i] = e[0]
            nodes[2 * i + 1] = e[1]

        nodes = nodes[:i * 2 + 1 + 1]

        nodes = np.unique(nodes)

        return nodes, edges

    def get_dependants_graph_subgraph(self, nodes, subedges):

        # parent_edges, children_edges = self.make_edges_map(subedges, nodes)

        edges = np.zeros_like(subedges)

        ix = int64(0)

        edges_visited = np.zeros(len(subedges), dtype=int64)
        n_visited = int64(0)
        for n in nodes:
            ix, n_visited = walk_children_(subedges, n, edges, ix, edges_visited, n_visited, self.node_types)

        edges = edges[:ix, :]
        edges[:ix, 3] = 2

        # nodes = np.zeros((0,1),int64)
        nodes = np.zeros(2 * len(edges), dtype=int64)

        for i, e in enumerate(edges):
            nodes[2 * i] = e[0]
            nodes[2 * i + 1] = e[1]

        # nodes = nodes[:i * 2+1+1]

        nodes = np.unique(nodes)

        return nodes, edges

    def get_anc_dep_graph(self, n):
        anc_nodes, anc_edges, deriv_dependencies = self.get_ancestor_graph(n)
        # print(anc_nodes)
        state_nodes = np.array([ancn for ancn in anc_nodes if self.node_types[ancn] >= 3], np.int64)
        # print('states: ', state_nodes)
        if len(state_nodes) > 0:
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
            if cyclic_dependency >= 0:
                print('cyclic dependency: ', cyclic_dependency)
                print('cyclic path: ', cyclic_path)
                # raise ValueError('cyclic dependency')
                return cyclic_dependency, cyclic_path

        return int64(-1), np.zeros((0,), dtype=int64)