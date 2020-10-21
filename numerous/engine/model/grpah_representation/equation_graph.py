from .graph import Graph
from numerous.engine.model.utils import NodeTypes


class EquationGraph(Graph):

    def variables(self):
        for n in self.node_map.values():
            if self.get(n, 'node_type') == NodeTypes.VAR:
                yield n


    def init_eg_graph(self):
        self.vars_assignments = {}
        self.vars_assignments_mappings = {}
        self.vars_mappings = {}


    def remove_chains(self):
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


    def create_assignments(self):
        from tqdm import tqdm

        ## generator ii  n i e for node
        for ii, n in tqdm(enumerate(self.get_where_attr('node_type', NodeTypes.EQUATION))):

            for i, e in self.get_edges_for_node(start_node=n):
                va = e[1].copy()
                if va in vars_assignments and len(vars_assignments[va]) > 1:

                    # Make new temp var
                    tmp_label = self.key_map[va] + '_tmp'
                    sv = self.get(e[1], 'scope_var')

                    # Create fake scope variables for tmp setvar
                    fake_sv = {}
                    svf = None
                    for i_, svi in tqdm(enumerate(scope_variables.values())):
                        if sv.set_var and svi.set_var == sv.set_var:
                            svf = TemporaryVar(svi, tmp_label)
                            fake_sv[d_u(svf.get_path_dot())] = svf

                    if not sv.set_var:
                        svf = TemporaryVar(sv, tmp_label)
                        fake_sv[d_u(svf.get_path_dot())] = svf

                    scope_variables.update(fake_sv)

                    tmp = self.add_node(key=tmp_label, node_type=NodeTypes.TMP, name=tmp_label, ast=None,
                                                  file='sum', label=tmp_label, ln=0,
                                                  ast_type=None, scope_var=svf, ignore_existing=False)
                    # Add temp var to Equation target

                    self.add_edge(n, tmp, e_type='target', arg_local=self.edges_attr['arg_local'][i[0]])
                    # Add temp var in var assignments

                    vars_assignments_mappings[va][(nix := vars_assignments[va].index(n))] = ':'
                    vars_assignments[va][nix] = tmp


    @classmethod
    def from_graph(cls, eg):
        eg.__class__ = cls
        eg.init_eg_graph()
        return eg
