from numba import int32, float64, int64, prange, njit, types
import numpy as np

numba_model_spec = [
    ('eq_count', int32),
    ('scope_vars_3d', float64[:, :, :]),
    ('state_idxs_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('deriv_idxs_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('differing_idxs_pos_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('differing_idxs_from_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('global_vars', float64[:]),
]


class NumbaModel:
    def __init__(self, eq_count, scope_vars_3d, state_idxs_3d, deriv_idxs_3d,
                 differing_idxs_pos_3d, differing_idxs_from_3d,
                 global_vars):
        self.eq_count = eq_count
        self.scope_vars_3d = scope_vars_3d
        self.state_idxs_3d = state_idxs_3d
        self.deriv_idxs_3d = deriv_idxs_3d
        self.differing_idxs_pos_3d = differing_idxs_pos_3d
        self.differing_idxs_from_3d = differing_idxs_from_3d
        self.global_vars = global_vars

    def update_states(self, state_values):
        self.scope_vars_3d[self.state_idxs_3d] = state_values

    def update_states_idx(self, state_value, idx_3d):
        self.scope_vars_3d[idx_3d] = state_value

    def get_derivatives(self):
        return self.scope_vars_3d[self.deriv_idxs_3d]

    def get_derivatives_idx(self, idx_3d):
        return self.scope_vars_3d[idx_3d]

    def compute(self):
        # if self.sum_mapping:
        #     sum_mappings(self.model.sum_idx, self.model.sum_mapped_idx, self.t_scope.scope_vars_3d,
        #                  self.model.sum_mapped)

        mapping_ = True
        prev_scope_vars_3d = self.scope_vars_3d.copy()
        while mapping_:
            self.scope_vars_3d[self.differing_idxs_pos_3d] = self.scope_vars_3d[
                self.differing_idxs_from_3d]
            self.compute_eq(self.scope_vars_3d)

            # if self.sum_mapping:
            #     sum_mappings(self.model.sum_idx, self.model.sum_mapped_idx, self.t_scope.scope_vars_3d,
            #                  self.model.sum_mapped)

            mapping_ = not np.all(np.abs(prev_scope_vars_3d - self.scope_vars_3d) < 1e-6)
            np.copyto(prev_scope_vars_3d, self.scope_vars_3d)

    def func(self, _t, y):
        # self.info["Number of Equation Calls"] += 1
        self.update_states(y)
        self.global_vars[0] = _t
        self.compute()

        return self.get_derivatives()


@njit
def sum_mappings(sum_idx, sum_mapped_idx, flat_var, sum_mapped):
    raise ValueError
    for i in prange(sum_idx.shape[0]):
        idx = sum_idx[i]
        slice_ = sum_mapped_idx[i]
        flat_var[idx] = np.sum(flat_var[sum_mapped[slice_[0]:slice_[1]]])
