from numerous.engine.variables import VariableType, SetOfVariables, Variable
from numerous.engine.model.graph_representation.utils import EdgeType
from numerous.engine.model.utils import NodeTypes
from numerous.utils.string_utils import d_u

from .graph import Graph, Node, Edge


class TemporaryVar(Variable):
    def __init__(self, id, svi, tag, set_var, set_var_ix):
        super().empty_variable()
        self.temporary_variable = True
        self.id = id
        self.tag = tag + id
        self.scope_var_id = svi.id
        self.set_var = set_var
        self.set_var_ix = set_var_ix
        self.value = svi.value
        self.type = VariableType.TMP_PARAMETER
        self.path = svi.path
        self.item = svi.item

    def get_path_dot(self):
        return self.tag


##should be inherited from SetOfVariables
class TemporarySetVar:

    def __init__(self, id, set_var):
        self.id = id
        self.tmp_vars = []
        self.type = VariableType.TMP_PARAMETER_SET
        self.set_var = set_var
        self.size = set_var.size
        self.variables = self.set_var.variables

    def get_size(self):
        return len(self.tmp_vars)

    def get_var_by_idx(self, i):
        return next(var for var in self.tmp_vars if var.set_var_ix == i)


class SumCount:
    def __init__(self):
        self.count = -1

    def get_sum(self):
        self.count += 1
        return f"sum_{self.count}"


new_sum = SumCount().get_sum


class MappingsGraph(Graph):

    def __init__(self, preallocate_items=1000):
        super().__init__(preallocate_items)
        self.vars_assignments = {}
        self.vars_mappings = {}
        self.vars_assignments_mappings = {}

    def variables(self):
        for n in self.node_map.values():
            if self.nodes[n].node_type == NodeTypes.VAR:
                yield n

    def remove_chains(self):
        for target in self.variables():
            # Get target
            target_edges_indcs_edge_type_target, target_edges_edge_type_target = self.get_edges_type_for_node_filter(
                end_node=target, val=EdgeType.TARGET)
            target_edges_indcs_edge_type_mapping, target_edges_edge_type_mapping = self.get_edges_type_for_node_filter(
                end_node=target, val=EdgeType.MAPPING)
            target_edges_indcs = target_edges_indcs_edge_type_target + target_edges_indcs_edge_type_mapping
            target_edges = target_edges_edge_type_target + target_edges_edge_type_mapping
            for edge, edge_ix in zip(target_edges, target_edges_indcs):

                if not target in self.vars_assignments:
                    self.vars_assignments[target] = []
                    self.vars_assignments_mappings[target] = []

                if self.edges_c[edge_ix].e_type == EdgeType.MAPPING:
                    self.vars_mappings[target] = (edge[0], self.edges_c[edge_ix].mappings)
                    self.remove_edge(edge_ix)

                self.vars_assignments[target].append(edge[0])
                self.vars_assignments_mappings[target].append(self.edges_c[edge_ix].mappings)

        for target in self.variables():
            target_edges_indcs, target_edges = self.get_edges_type_for_node_filter(end_node=target,
                                                                                   val=[EdgeType.TARGET, EdgeType.MAPPING])
            if target in self.vars_assignments and len(self.vars_assignments[target]) > 1:
                for edge_ix in target_edges_indcs:
                    self.remove_edge(edge_ix)

    def create_assignments(self, variables):
        from tqdm import tqdm
        temp_variables = {}
        for ii, n in tqdm(enumerate(self.get_where_node_attr('node_type', NodeTypes.EQUATION)),
                          desc="Creating assigments"):
            for i, e in self.get_edges_for_node(start_node=n):
                va = e[1].copy()
                if va in self.vars_assignments and len(self.vars_assignments[va]) > 1:
                    # Make new temp var
                    sv = self.nodes[e[1]].scope_var
                    tmp_key = sv.tag + str(self.key_map[va]) + '_tmp'
                    tmp_label = sv.tag + str(self.key_map[va]) + '_label' + '_tmp'
                    # Create fake scope variables for tmp setvar
                    fake_sv = {}
                    svf = None
                    if isinstance(sv, SetOfVariables):
                        tmp_var_counter = 0
                        tsv = TemporarySetVar(tmp_key, sv)
                        for svi in sv.variables.values():
                            tmp_var_counter += 1
                            svf = TemporaryVar('tmp_var_' + str(tmp_var_counter), svi,
                                               svi.tag + '_' + str(sv.id), svi.set_var, svi.set_var_ix)
                            tsv.tmp_vars.append(svf)
                        fake_sv[tsv.id] = tsv
                    else:
                        svf = TemporaryVar(d_u(tmp_key), sv, tmp_key, None, None)
                        fake_sv[d_u(svf.get_path_dot())] = svf

                    temp_variables.update(fake_sv)

                    tmp = self.add_node(Node(key=tmp_key, node_type=NodeTypes.TMP, name=tmp_key,
                                             file='sum', label=tmp_label, ln=0, scope_var=svf), ignore_existing=False)
                    # Add temp var to Equation target

                    self.add_edge(Edge(n, tmp, e_type=EdgeType.TARGET, arg_local=self.edges_c[i[0]].arg_local))
                    # Add temp var in var assignments

                    self.vars_assignments_mappings[va][(nix := self.vars_assignments[va].index(n))] = ':'
                    self.vars_assignments[va][nix] = tmp
        return temp_variables

    def add_mappings(self):
        for a, vals in self.vars_assignments.items():
            if len(vals) > 1:
                ns = new_sum()
                nsn = self.add_node(Node(key=ns, node_type=NodeTypes.SUM, name=ns, file='sum',
                                         label=ns,
                                         ln=0, ast_type=None))
                self.add_edge(Edge(start=nsn, end=a, e_type=EdgeType.TARGET))
                for v, mappings in zip(vals, self.vars_assignments_mappings[a]):
                    self.add_edge(Edge(start=v, end=nsn, e_type=EdgeType.VALUE, mappings=mappings))

            elif a in self.vars_mappings:
                ns = new_sum()
                nsn = self.add_node(Node(key=ns, node_type=NodeTypes.SUM, name=ns, file='sum',
                                         label=ns,
                                         ln=0, ast_type=None))
                self.add_edge(Edge(start=nsn, end=a, e_type=EdgeType.TARGET))

                self.add_edge(Edge(start=self.vars_mappings[a][0], end=nsn,
                                   e_type=EdgeType.VALUE, mappings=self.vars_mappings[a][1]))

    @classmethod
    def from_graph(cls, eg):
        eg.__class__ = cls
        eg.vars_assignments = {}
        eg.vars_mappings = {}
        eg.vars_assignments_mappings = {}
        return eg
