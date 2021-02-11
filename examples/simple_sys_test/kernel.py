from numba import njit, carray, float64, float32
import numpy as np
def EQ_efd619f6_59a1_11eb_a53f_1d96dbf0e512system_SET_simples_mechanics(s_x_dot
    , s_x, s_k):
    s_x_dot = s_k + float64(0) * s_x
    return s_x_dot


kernel_variables = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.2])


def global_kernel(states):
    np.put(kernel_variables, [1, 4], states)
    for i in range(2):
        kernel_variables[0 + 3 * i] = (
            EQ_efd619f6_59a1_11eb_a53f_1d96dbf0e512system_SET_simples_mechanics
            (kernel_variables[0 + 3 * i], kernel_variables[1 + 3 * i],
            kernel_variables[2 + 3 * i]))
    return np.take(kernel_variables, [0, 3])
