from .graph import Graph
from numerous.engine.model.utils import NodeTypes


class EquationGraph(Graph):

    def variables(self):
        for n in self.node_map.values():
            if self.get(n, 'node_type') == NodeTypes.VAR:
                yield n

    def remove_chains(self):
        vars_assignments = {}
        vars_assignments_mappings = {}
        vars_mappings = {}

        for target in self.variables():
            # Get target
            target_edges_indcs, target_edges = self.get_edges_for_node_filter(end_node=target, attr='e_type',
                                                                              val=['target', 'mapping'])
            for edge, edge_ix in zip(target_edges, target_edges_indcs):

                if not target in vars_assignments:
                    vars_assignments[target] = []
                    vars_assignments_mappings[target] = []

                if self.edges_attr['e_type'][edge_ix] == 'mapping':
                    vars_mappings[target] = (edge[0], self.edges_attr['mappings'][edge_ix])
                    self.remove_edge(edge_ix)

                vars_assignments[target].append(edge[0])
                vars_assignments_mappings[target].append(self.edges_attr['mappings'][edge_ix])

            if target in vars_assignments and len(vars_assignments[target]) > 1:
                for edge_ix in target_edges_indcs:
                    self.remove_edge(edge_ix)

    @classmethod
    def from_graph(cls, eg):
        eg.__class__ = cls
        return eg
