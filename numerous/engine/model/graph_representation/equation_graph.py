from numerous import VariableType
from string_utils import d_u
from .graph import Graph
from numerous.engine.model.utils import NodeTypes


class TemporaryVar():
    tmp_var_counter = 0

    def __init__(self, svi, tmp_label):
        TemporaryVar.tmp_var_counter += 1
        self.id = 'tmp_var_' + str(TemporaryVar.tmp_var_counter) if svi.set_var else d_u(tmp_label)
        self.tag = svi.tag + '_' + self.id if svi.set_var else tmp_label
        self.set_namespace = svi.set_namespace
        self.parent_scope_id = svi.parent_scope_id
        self.set_var = tmp_label if svi.set_var else None
        self.set_var_ix = svi.set_var_ix if svi.set_var else None
        self.value = svi.value
        self.type = VariableType.PARAMETER
        self.path = svi.path

    def get_path_dot(self):
        return self.tag

class SumCount:
    def __init__(self):
        self.count = -1

    def get_sum(self):
        self.count += 1
        return f"sum_{self.count}"


new_sum = SumCount().get_sum


class EquationGraph(Graph):

    def __init__(self, preallocate_items=1000):
        super().__init__(preallocate_items)
        self.vars_assignments = {}
        self.vars_mappings = {}
        self.vars_assignments_mappings = {}

    def variables(self):
        for n in self.node_map.values():
            if self.get(n, 'node_type') == NodeTypes.VAR:
                yield n

    def remove_chains(self):
        for target in self.variables():
            # Get target
            target_edges_indcs, target_edges = self.get_edges_for_node_filter(end_node=target, attr='e_type',
                                                                              val=['target', 'mapping'])
            for edge, edge_ix in zip(target_edges, target_edges_indcs):

                if not target in self.vars_assignments:
                    self.vars_assignments[target] = []
                    self.vars_assignments_mappings[target] = []

                if self.edges_attr['e_type'][edge_ix] == 'mapping':
                    self.vars_mappings[target] = (edge[0], self.edges_attr['mappings'][edge_ix])
                    self.remove_edge(edge_ix)

                self.vars_assignments[target].append(edge[0])
                self.vars_assignments_mappings[target].append(self.edges_attr['mappings'][edge_ix])

            if target in self.vars_assignments and len(self.vars_assignments[target]) > 1:
                for edge_ix in target_edges_indcs:
                    self.remove_edge(edge_ix)

    def create_assignments(self, scope_variables):
        from tqdm import tqdm

        ## generator ii  n i e for node
        for ii, n in tqdm(enumerate(self.get_where_attr('node_type', NodeTypes.EQUATION))):

            for i, e in self.get_edges_for_node(start_node=n):
                va = e[1].copy()
                if va in self.vars_assignments and len(self.vars_assignments[va]) > 1:

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

                    self.vars_assignments_mappings[va][(nix := self.vars_assignments[va].index(n))] = ':'
                    self.vars_assignments[va][nix] = tmp

    def add_mappings(self):
        for a, vals in self.vars_assignments.items():
            if len(vals) > 1:
                ns = new_sum()
                nsn = self.add_node(key=ns, node_type=NodeTypes.SUM, name=ns, ast=None, file='sum',
                                    label=ns,
                                    ln=0, ast_type=None)
                self.add_edge(nsn, a, e_type='target')
                for v, mappings in zip(vals, self.vars_assignments_mappings[a]):
                    self.add_edge(v, nsn, e_type='value', mappings=mappings)

            elif a in self.vars_mappings:
                ns = new_sum()
                nsn = self.add_node(key=ns, node_type=NodeTypes.SUM, name=ns, ast=None, file='sum',
                                    label=ns,
                                    ln=0, ast_type=None)
                self.add_edge(nsn, a, e_type='target')

                self.add_edge(self.vars_mappings[a][0], nsn, e_type='value', mappings=self.vars_mappings[a][1])

    @classmethod
    def from_graph(cls, eg):
        eg.__class__ = cls
        eg.vars_assignments = {}
        eg.vars_mappings = {}
        eg.vars_assignments_mappings = {}
        return eg
