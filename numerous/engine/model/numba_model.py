from numba import int32, float64, int64, boolean, prange, njit
import numpy as np

numba_model_spec = [
    ('eq_count', int32),
    ('sum_idx', int32[:]),
    ('sum_mapped_idx', int32[:]),
    ('sum_mapped', float64[:]),
    ('sum_mapping', boolean),
    ('compiled_eq_idxs', int32[:]),
    ('index_helper', int64[:]),
    ('length', int64[:]),
    ('flat_scope_idx_from', int64[:]),
    ('flat_scope_idx_from_idx_1', int64[:]),
    ('flat_scope_idx_from_idx_2', int64[:]),
    ('flat_scope_idx', int64[:]),
    ('flat_scope_idx_idx_1', int64[:]),
    ('flat_scope_idx_idx_2', int64[:]),
    ('flat_var', float64[:]),
    ('state_idx', int32[:]),
    ('deriv_idx', int32[:]),
    ('global_vars', float64[:]),
    ('scope_variables_2d', float64[:, :, :]),
]


class NumbaModel:
    def __init__(self,eq_count,
                 sum_idx, sum_mapped_idx,
                 sum_mapped, compiled_eq_idxs,
                 index_helper, length, flat_scope_idx_from,
                 flat_scope_idx_from_idx_1, flat_scope_idx_from_idx_2,
                 flat_scope_idx, flat_scope_idx_idx_1, flat_scope_idx_idx_2,
                 flat_scope_var, state_idx, deriv_idx, global_vars, scope_variables_2d):
        self.eq_count = eq_count
        self.sum_idx = sum_idx
        self.sum_mapped_idx = sum_mapped_idx
        self.sum_mapped = sum_mapped
        self.sum_mapping = sum_idx.size != 0
        self.compiled_eq_idxs = compiled_eq_idxs
        self.index_helper = index_helper
        self.length = length
        self.flat_scope_idx_from = flat_scope_idx_from
        self.flat_scope_idx_from_idx_1 = flat_scope_idx_from_idx_1
        self.flat_scope_idx_from_idx_2 = flat_scope_idx_from_idx_2
        self.flat_scope_idx = flat_scope_idx
        self.flat_scope_idx_idx_1 = flat_scope_idx_idx_1
        self.flat_scope_idx_idx_2 = flat_scope_idx_idx_2
        self.flat_var = flat_scope_var
        self.state_idx = state_idx
        self.deriv_idx = deriv_idx
        self.global_vars = global_vars
        self.scope_variables_2d = scope_variables_2d

    def update_states(self, state_values):
        self.flat_var[self.state_idx] = state_values

    def update_states_idx(self, state_value, idx):
        self.flat_var[idx] = state_value

    def get_derivatives(self):
        return self.flat_var[self.deriv_idx]

    def get_derivatives_idx(self, idx):
        return self.flat_var[idx]

    def compute(self):
        # if self.sum_mapping:
        #     sum_mappings(self.sum_idx, self.sum_mapped_idx,
        #                  self.flat_var,
        #                  self.sum_mapped)
        mapping_ = True

        b1 = np.copy(self.flat_var)
        while mapping_:
            mapping_from(self.compiled_eq_idxs, self.index_helper,
                         self.scope_variables_2d,
                         self.length, self.flat_var,
                         self.flat_scope_idx_from,
                         self.flat_scope_idx_from_idx_1,
                         self.flat_scope_idx_from_idx_2)

            self.compute_eq(self.scope_variables_2d)

            mapping_to(self.compiled_eq_idxs, self.flat_var,
                       self.flat_scope_idx,
                       self.scope_variables_2d,
                       self.index_helper, self.length,
                       self.flat_scope_idx_idx_1, self.flat_scope_idx_idx_2)

            # if self.sum_mapping:
            #     sum_mappings(self.sum_idx, self.sum_mapped_idx,
            #                  self.flat_var,
            #                  self.sum_mapped)

            mapping_ = not np.all(np.abs(b1 - self.flat_var) < 1e-6)
            b1 = np.copy(self.flat_var)

    def func(self, _t, y):
        self.update_states(y)
        self.global_vars[0] = _t
        self.compute()

        return self.get_derivatives()


@njit
def mapping_to(compiled_eq_idxs, flat_var, flat_scope_idx, scope_variables_2d, index_helper, length, id1, id2):
    for i in prange(compiled_eq_idxs.shape[0]):
        eq_idx = compiled_eq_idxs[i]
        flat_var[flat_scope_idx[id1[i]:id2[i]]] = \
            scope_variables_2d[eq_idx][index_helper[i]][:length[i]]


@njit
def mapping_from(compiled_eq_idxs, index_helper, scope_variables_2d, length, flat_var, flat_scope_idx_from, id1,
                 id2):
    for i in prange(compiled_eq_idxs.shape[0]):
        eq_idx = compiled_eq_idxs[i]
        scope_variables_2d[eq_idx][index_helper[i]][:length[i]] \
            = flat_var[flat_scope_idx_from[id1[i]:id2[i]]]


@njit
def sum_mappings(sum_idx, sum_mapped_idx, flat_var, sum_mapped):
    for i in prange(sum_idx.shape[0]):
        idx = sum_idx[i]
        slice_ = sum_mapped_idx[i]
        flat_var[idx] = np.sum(flat_var[sum_mapped[slice_[0]:slice_[1]]])
