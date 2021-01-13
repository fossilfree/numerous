from numba import int32, float64, boolean, int64, njit, types, typed
import numpy as np

# key and value types
kv_ty = (types.unicode_type, float64)

numba_model_spec = [
    ('start_time', int32),
    ('number_of_timesteps', int32),
    ('deriv_idx', int64[:]),
    ('state_idx', int64[:]),
    ('init_vars', float64[:]),
    ('global_vars', float64[:]),
    ('historian_ix', int64),
    ('historian_data', float64[:, :]),
    ('path_variables', types.DictType(*kv_ty)),
    ('path_keys', types.ListType(types.unicode_type)),
    ('historian_max_size', int64),
    ('in_memory_history_correction', boolean),
]


@njit
def step_aproximation(t, time_array, data_array):
    idx = np.searchsorted(time_array, t, side='right') - 1
    return data_array[idx]


class CompiledModel:
    def __init__(self, init_vars,deriv_idx,state_idx,
                 global_vars, number_of_timesteps, start_time, historian_max_size,
                 in_memory_history_correction):
        self.global_vars = global_vars
        self.deriv_idx =deriv_idx
        self.state_idx = state_idx
        self.path_variables = typed.Dict.empty(*kv_ty)
        self.path_keys = typed.List.empty_list(types.unicode_type)
        self.number_of_timesteps = number_of_timesteps
        self.start_time = start_time
        self.historian_ix = 0
        self.init_vars=init_vars
        self.historian_max_size = historian_max_size
        self.in_memory_history_correction = in_memory_history_correction
        self.historian_data = np.empty((len(init_vars) + 1, self.historian_max_size- self.in_memory_history_correction), dtype=np.float64)
        self.historian_data.fill(np.nan)


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

    # def map_external_data(self, t):
    #     for i in range(self.number_of_external_mappings):
    #         df_indx = self.external_df_idx[i][0]
    #         var_idx = self.external_df_idx[i][1]
    #         self.scope_vars_3d[self.external_idx_3d[0][i]][self.external_idx_3d[1][i]][
    #             self.external_idx_3d[2][i]] = np.interp(t, self.external_mappings_time[df_indx],
    #                                                     self.external_mappings_numpy[df_indx, :, var_idx]) if \
    #             self.approximation_type[i] else \
    #             step_aproximation(t, self.external_mappings_time[df_indx],
    #                               self.external_mappings_numpy[df_indx, :, var_idx])

    # def update_external_data(self, external_mappings_numpy, external_mappings_time):
    #     self.external_mappings_time = external_mappings_time
    #     self.external_mappings_numpy = external_mappings_numpy
    #
    # def is_external_data_update_needed(self, t):
    #     if self.is_external_data and t > self.max_external_t:
    #         return True
    #     return False

    def get_states(self):
        return self.read_variables()[self.state_idx]

    def is_store_required(self):
        if self.historian_ix >= self.historian_max_size:
            return True
        return False

    def historian_reinit(self):
        self.historian_data = np.empty(
            (len(self.init_vars) + 1, self.historian_max_size - self.in_memory_history_correction), dtype=np.float64)
        self.historian_data.fill(np.nan)
        self.historian_ix = 0

    def historian_update(self, time: np.float64) -> None:
        ix = self.historian_ix
        self.historian_data[0][ix] = time
        self.historian_data[1:,ix]= self.read_variables()
        self.historian_ix += 1



    # def run_callbacks_with_updates(self, time: int) -> None:
    #     '''
    #     Updates all the values of all Variable instances stored in
    #     `self.variables` with the values stored in `self.scope_vars_3d`.
    #     '''
    #     if self.callbacks:
    #         for key, j in zip(self.path_keys,
    #                           self.var_idxs_pos_3d_helper_callbacks):
    #             self.path_variables[key] \
    #                 = self.scope_vars_3d[self.var_idxs_pos_3d[0][j]][self.var_idxs_pos_3d[1][j]][
    #                 self.var_idxs_pos_3d[2][j]]
    #
    #         self.run_callbacks(time)
    #
    #         for key, j in zip(self.path_keys,
    #                           self.var_idxs_pos_3d_helper_callbacks):
    #             self.scope_vars_3d[self.var_idxs_pos_3d[0][j]][self.var_idxs_pos_3d[1][j]][self.var_idxs_pos_3d[2][j]] \
    #                 = self.path_variables[key]


    def get_g(self, t, yold, y, dt, order, a, af):
        f = self.func(t, y)
        _sum = np.zeros_like(y)
        for i in range(order):
            _sum = _sum + a[order - 1][i] * yold[i, :]
        g = y + _sum - af[order - 1] * dt * f
        return np.ascontiguousarray(g), np.ascontiguousarray(f)


    def func(self, _t, y):
        self.global_vars[0] = _t
        deriv = self.compiled_compute(y)
        return deriv



