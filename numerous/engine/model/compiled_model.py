from types import FunctionType

from numba import int32, float64, boolean, int64, njit, types, typed
import numpy as np
import numpy.typing as npt

try:
    FEPS = np.finfo(1.0).eps
except AttributeError:
    FEPS = 2.220446049250313e-16


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
    ('historian_max_size', int64),
    ('in_memory_history_correction', boolean),
    ('external_mappings_time', float64[:, :]),
    ('number_of_external_mappings', int64),
    ('external_mappings_numpy', float64[:, :, :]),
    ('external_df_idx', int64[:, :]),
    ('approximation_type', boolean[:]),
    ('is_external_data', boolean),
    ('max_external_t', float64),
    ('min_external_t', float64),
    ('external_idx', int64[:]),
    ('previous_external_mappings', float64[:]),
    ('post_step', types.ListType(types.FunctionType(types.void())))
]


@njit
def closest_time_idx(t, time_array):
    return np.searchsorted(time_array, t + 100 * FEPS, side='right') - 1


@njit
def step_approximation(t, time_array, data_array):
    idx = closest_time_idx(t, time_array)
    return data_array[idx]


class CompiledModel:
    def __init__(self, init_vars, deriv_idx, state_idx,
                 global_vars, number_of_timesteps, start_time, historian_max_size,
                 external_mappings_time, number_of_external_mappings,
                 external_mappings_numpy, external_df_idx, interpolation_info,
                 is_external_data, t_max, t_min, external_idx, post_step):
        self.external_idx = external_idx
        self.external_mappings_time = external_mappings_time
        self.number_of_external_mappings = number_of_external_mappings
        self.external_mappings_numpy = external_mappings_numpy
        self.post_step = post_step
        self.external_df_idx = external_df_idx
        self.approximation_type = interpolation_info
        self.is_external_data = is_external_data
        self.max_external_t = t_max
        self.min_external_t = t_min
        self.global_vars = global_vars
        self.deriv_idx = deriv_idx
        self.state_idx = state_idx
        self.number_of_timesteps = number_of_timesteps
        self.start_time = start_time
        self.historian_ix = 0
        self.init_vars = init_vars
        self.historian_max_size = historian_max_size
        self.historian_data = np.empty(
            (len(init_vars) + 1, self.historian_max_size), dtype=np.float64)
        self.historian_data.fill(np.nan)
        self.previous_external_mappings = np.zeros(self.number_of_external_mappings)

    def vectorized_full_jacobian(self, t, y, dt):
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

    def map_external_data(self, t):

        if not self.is_external_data:
            return

        next_values = np.zeros(self.number_of_external_mappings)
        for i in range(self.number_of_external_mappings):

            df_indx = self.external_df_idx[i][0]
            var_idx = self.external_df_idx[i][1]
            value = np.interp(t, self.external_mappings_time[df_indx],
                              self.external_mappings_numpy[df_indx, :, var_idx]) if \
                self.approximation_type[i] else \
                step_approximation(t, self.external_mappings_time[df_indx],
                                   self.external_mappings_numpy[df_indx, :, var_idx])
            next_values[i] = value

        if np.array_equal(next_values, self.previous_external_mappings):  # No need to update if no values were changed
            return

        self.previous_external_mappings = next_values

        for i, value in enumerate(next_values):
            self.write_variables(value, self.external_idx[i])

        self.func(t, self.get_states())

    def run_post_step(self):
        for post_step_ in self.post_step:
            post_step_()

    def update_external_data(self, external_mappings_numpy, external_mappings_time, max_external_t, min_external_t):
        self.external_mappings_time = external_mappings_time
        self.external_mappings_numpy = external_mappings_numpy
        self.max_external_t = max_external_t
        self.min_external_t = min_external_t

    def is_external_data_update_needed(self, t):
        if self.is_external_data and (t > self.max_external_t or t < self.min_external_t):
            return True
        return False

    def get_states(self):
        return self.read_variables()[self.state_idx]

    def set_states(self, states: npt.ArrayLike) -> None:
        for i in range(len(states)):
            self.write_variables(states[i], self.state_idx[i])

    def is_store_required(self):
        if self.historian_ix >= self.historian_max_size:
            return True
        return False

    def historian_reinit(self):
        self.historian_data = np.empty(
            (len(self.init_vars) + 1, self.historian_max_size), dtype=np.float64)
        self.historian_data.fill(np.nan)
        self.historian_ix = 0

    def historian_update(self, time: np.float64) -> None:
        ix = self.historian_ix
        self.historian_data[0][ix] = time
        self.historian_data[1:, ix] = self.read_variables()
        self.historian_ix += 1

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
