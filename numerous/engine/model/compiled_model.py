import math

from numba import int32, float64, boolean, int64, prange, njit, types, typed
import numpy as np

# key and value types
kv_ty = (types.unicode_type, float64)

numba_model_spec = [
    ('var_idxs_pos_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('var_idxs_pos_3d_helper_callbacks', int64[:]),
    ('var_idxs_historian_3d', int64[:]),
    ('start_time', int32),
    ('eq_count', int32),
    ('number_of_timesteps', int32),
    ('number_of_states', int32),
    ('number_of_mappings', int32),
    ('scope_vars_3d', float64[:, :, :]),
    ('state_idxs_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('deriv_idxs_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('differing_idxs_pos_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('differing_idxs_from_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('num_uses_per_eq', int64[:]),
    ('sum_idxs_pos_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('sum_idxs_sum_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('sum_slice_idxs', int64[:]),
    ('sum_slice_idxs_len', int64[:]),
    ('sum_mapping', boolean),
    ('callbacks', boolean),
    ('global_vars', float64[:]),
    ('historian_ix', int64),
    ('historian_data', float64[:, :]),
    ('path_variables', types.DictType(*kv_ty)),
    ('path_keys', types.ListType(types.unicode_type)),
    ('mapped_variables_array', int64[:, :]),
    ('external_mappings_time', float64[:, :]),
    ('number_of_external_mappings', int64),
    ('external_idx_3d', types.Tuple((int64[:], int64[:], int64[:]))),
    ('external_mappings_numpy', float64[:, :, :]),
    ('external_df_idx', int64[:, :]),
    ('approximation_type', boolean[:])
]


@njit
def step_aproximation(t, time_array, data_array):
    idx = np.searchsorted(time_array, t, side='left')
    return data_array[idx]


class CompiledModel:
    def __init__(self, var_idxs_pos_3d, var_idxs_pos_3d_helper_callbacks, var_idxs_historian_3d, eq_count,
                 number_of_states,
                 number_of_mappings,
                 scope_vars_3d, state_idxs_3d, deriv_idxs_3d,
                 differing_idxs_pos_3d, differing_idxs_from_3d, num_uses_per_eq,
                 sum_idxs_pos_3d, sum_idxs_sum_3d, sum_slice_idxs, sum_slice_idxs_len, sum_mapping, callbacks,
                 global_vars, number_of_timesteps, start_time, mapped_variables_array,
                 external_mappings_time, number_of_external_mappings, external_idx_3d, external_mappings_numpy,
                 external_df_idx, approximation_type):

        self.external_idx_3d = external_idx_3d
        self.external_df_idx = external_df_idx
        self.external_mappings_numpy = external_mappings_numpy
        self.external_mappings_time = external_mappings_time
        self.number_of_external_mappings = number_of_external_mappings
        self.var_idxs_pos_3d = var_idxs_pos_3d
        self.var_idxs_pos_3d_helper_callbacks = var_idxs_pos_3d_helper_callbacks
        self.var_idxs_historian_3d = var_idxs_historian_3d
        self.eq_count = eq_count
        self.callbacks = callbacks
        self.number_of_states = number_of_states
        self.scope_vars_3d = scope_vars_3d
        self.state_idxs_3d = state_idxs_3d
        self.deriv_idxs_3d = deriv_idxs_3d
        self.differing_idxs_pos_3d = differing_idxs_pos_3d
        self.differing_idxs_from_3d = differing_idxs_from_3d
        self.num_uses_per_eq = num_uses_per_eq
        self.number_of_mappings = number_of_mappings
        self.sum_idxs_pos_3d = sum_idxs_pos_3d
        self.sum_idxs_sum_3d = sum_idxs_sum_3d
        self.sum_slice_idxs = sum_slice_idxs
        self.sum_slice_idxs_len = sum_slice_idxs_len
        self.sum_mapping = sum_mapping
        self.global_vars = global_vars
        self.path_variables = typed.Dict.empty(*kv_ty)
        self.path_keys = typed.List.empty_list(types.unicode_type)
        self.number_of_timesteps = number_of_timesteps
        self.start_time = start_time
        self.historian_ix = 0
        self.approximation_type = approximation_type
        self.historian_data = np.empty((len(var_idxs_historian_3d) + 1, number_of_timesteps), dtype=np.float64)
        self.historian_data.fill(np.nan)
        self.mapped_variables_array = mapped_variables_array
        ##Function is genrated in model.py contains creation and initialization of all callback related variables

    def update_states(self, state_values):
        for i in range(self.number_of_states):
            self.scope_vars_3d[self.state_idxs_3d[0][i]][self.state_idxs_3d[1][i]][self.state_idxs_3d[2][i]] \
                = state_values[i]

    def update_states_idx(self, state_value, idx_3d):
        self.scope_vars_3d[idx_3d] = state_value

    def get_derivatives(self):
        result = []
        for i in range(self.number_of_states):
            result.append(
                self.scope_vars_3d[self.deriv_idxs_3d[0][i]][self.deriv_idxs_3d[1][i]][self.deriv_idxs_3d[2][i]])
        return np.array(result, dtype=np.float64)

    def get_mapped_variables(self, scope_3d):
        result = []
        for i in range(self.number_of_mappings):
            result.append(
                scope_3d[self.differing_idxs_from_3d[0][i]][self.differing_idxs_from_3d[1][i]]
                [self.differing_idxs_from_3d[2][i]])
        return np.array(result, dtype=np.float64)

    def get_states(self):
        result = []
        for i in range(self.number_of_states):
            result.append(
                self.scope_vars_3d[self.state_idxs_3d[0][i]][self.state_idxs_3d[1][i]][self.state_idxs_3d[2][i]])

        return np.array(result, dtype=np.float64)

    def map_external_data(self, t):
        for i in range(self.number_of_external_mappings):
            df_indx = self.external_df_idx[i][0]
            var_idx = self.external_df_idx[i][1]
            self.scope_vars_3d[self.external_idx_3d[0][i]][self.external_idx_3d[1][i]][
                self.external_idx_3d[2][i]] = np.interp(t, self.external_mappings_time[df_indx],
                                                        self.external_mappings_numpy[df_indx, :, var_idx]) if \
            self.approximation_type[i] else \
                step_aproximation(t, self.external_mappings_time[df_indx],
                                  self.external_mappings_numpy[df_indx, :, var_idx])

    def historian_update(self, time: np.float64) -> None:
        ix = self.historian_ix
        varix = 1
        self.historian_data[0][ix] = time
        # for j in self.var_idxs_pos_3d_helper:
        for j in self.var_idxs_historian_3d:
            self.historian_data[varix][ix] = self.scope_vars_3d[self.var_idxs_pos_3d[0][j]][self.var_idxs_pos_3d[1][j]][
                self.var_idxs_pos_3d[2][j]]
            varix += 1
        self.historian_ix += 1

    def run_callbacks_with_updates(self, time: int) -> None:
        '''
        Updates all the values of all Variable instances stored in
        `self.variables` with the values stored in `self.scope_vars_3d`.
        '''
        if self.callbacks:
            for key, j in zip(self.path_keys,
                              self.var_idxs_pos_3d_helper_callbacks):
                self.path_variables[key] \
                    = self.scope_vars_3d[self.var_idxs_pos_3d[0][j]][self.var_idxs_pos_3d[1][j]][
                    self.var_idxs_pos_3d[2][j]]

            self.run_callbacks(time)

            for key, j in zip(self.path_keys,
                              self.var_idxs_pos_3d_helper_callbacks):
                self.scope_vars_3d[self.var_idxs_pos_3d[0][j]][self.var_idxs_pos_3d[1][j]][self.var_idxs_pos_3d[2][j]] \
                    = self.path_variables[key]

    def get_derivatives_idx(self, idx_3d):
        return self.scope_vars_3d[idx_3d]

    def vectorizedfulljacobian(self, t, y, dt):
        h = 1e-8
        y_perm = y + h * np.diag(np.ones(len(y)))

        f = self.func(t, y)
        f_h = np.zeros_like(y_perm)
        for i in range(y_perm.shape[0]):
            y_i = y_perm[i, :]
            f_h[i, :] = self.func(t, y_i)

        diff = f_h - f
        diff /= h
        jac = diff.T
        return np.ascontiguousarray(jac)

    def get_g(self, t, yold, y, dt, order, a, af):
        f = self.func(t, y)
        _sum = np.zeros_like(y)
        for i in range(order):
            _sum = _sum + a[order - 1][i] * yold[i, :]
        g = y + _sum - af[order - 1] * dt * f
        return np.ascontiguousarray(g), np.ascontiguousarray(f)

    def compute(self, only_propagate_mappings=False):
        if self.sum_mapping:
            sum_mappings(self.sum_idxs_pos_3d, self.sum_idxs_sum_3d,
                         self.sum_slice_idxs, self.scope_vars_3d, self.sum_slice_idxs_len)

        mapping_ = True
        prev_scope_vars_3d = self.scope_vars_3d.copy()
        start_scope_vars_3d = self.scope_vars_3d.copy()
        itermax = 20
        it = 0
        ix_hist_old = np.argwhere(self.var_idxs_historian_3d == self.var_idxs_pos_3d_helper_callbacks[0])
        while mapping_:
            for i in range(self.number_of_mappings):
                self.scope_vars_3d[self.differing_idxs_pos_3d[0][i]][self.differing_idxs_pos_3d[1][i]][
                    self.differing_idxs_pos_3d[2][i]] = self.scope_vars_3d[
                    self.differing_idxs_from_3d[0][i]][self.differing_idxs_from_3d[1][i]][
                    self.differing_idxs_from_3d[2][i]]
            if not only_propagate_mappings:
                self.compute_eq(self.scope_vars_3d)

            if self.sum_mapping:
                sum_mappings(self.sum_idxs_pos_3d, self.sum_idxs_sum_3d,
                             self.sum_slice_idxs, self.scope_vars_3d, self.sum_slice_idxs_len)

            mapping_ = not np.all(np.abs(prev_scope_vars_3d - self.scope_vars_3d) < 1e-6)

            it += 1
            if it > itermax:
                raise Exception("maximum number of iterations has been reached")
            prev_scope_vars_3d = np.copy(self.scope_vars_3d)

    def func(self, _t, y):
        # self.info["Number of Equation Calls"] += 1
        self.update_states(y)
        self.global_vars[0] = _t
        self.compute()
        return self.get_derivatives()


@njit
def sum_mappings(sum_idxs_pos_3d, sum_idxs_sum_3d,
                 sum_slice_idxs, scope_vars_3d, sum_slice_idxs_len):
    idx_sum = 0
    for i, len_ in enumerate(sum_slice_idxs_len):
        sum_ = 0
        for j in sum_slice_idxs[idx_sum:idx_sum + len_]:
            sum_ += scope_vars_3d[sum_idxs_sum_3d[0][j]][sum_idxs_sum_3d[1][j]][sum_idxs_sum_3d[2][j]]
        idx_sum += len_
        scope_vars_3d[sum_idxs_pos_3d[0][i]][sum_idxs_pos_3d[1][i]][sum_idxs_pos_3d[2][i]] = sum_
