from numba import njit, carray, float64, float32
import numpy as np
@njit
def DampenedOscillator_eval(s_v, s_k, s_x, s_c):
    s_x_dot = s_v
    s_a = -s_k * s_x - s_c * s_v
    s_v_dot = s_a
    return s_x_dot, s_a, s_v_dot


def DampenedOscillator_eval_llvm(s_v, s_k, s_x, s_c, s_x_dot, s_a, s_v_dot):
    carray(s_x_dot, (1,))[0] = s_v
    carray(s_a, (1,))[0] = -s_k * s_x - s_c * s_v
    carray(s_v_dot, (1,))[0] = carray(s_a, (1,))[0]


@njit
def Spring_Equation_eval(s_x1, s_x2, s_k):
    dx = s_x1 - s_x2
    s_c = s_k
    F = np.abs(dx) * s_c
    s_F1 = -F if s_x1 > s_x2 else F
    s_F2 = -s_F1
    return s_c, s_F1, s_F2


def Spring_Equation_eval_llvm(s_x1, s_x2, s_k, s_c, s_F1, s_F2):
    dx = s_x1 - s_x2
    carray(s_c, (1,))[0] = s_k
    F = np.abs(dx) * carray(s_c, (1,))[0]
    carray(s_F1, (1,))[0] = -F if s_x1 > s_x2 else F
    carray(s_F2, (1,))[0] = -carray(s_F1, (1,))[0]


def kernel_nojit(variables, y):
    system_SET_couplings_mechanics_k = variables[[10, 5]]
    system_SET_oscillators_mechanics_k = variables[[9, 6]]
    system_SET_oscillators_mechanics_v = variables[[0, 2]]
    system_SET_oscillators_mechanics_x = variables[[1, 3]]
    system_SET_oscillators_mechanics_c = variables[[8, 7]]
    system_SET_oscillators_mechanics_v[0], system_SET_oscillators_mechanics_x[0], system_SET_oscillators_mechanics_v[1], system_SET_oscillators_mechanics_x[1] = y
    system_SET_couplings_mechanics_x2 = np.empty(2)
    system_SET_couplings_mechanics_x1 = np.empty(2)
    system_SET_couplings_mechanics_c = np.empty(2)
    system_SET_couplings_mechanics_F1 = np.empty(2)
    system_SET_couplings_mechanics_F2 = np.empty(2)
    system_SET_oscillators_mechanics_x_dot = np.empty(2)
    system_SET_oscillators_mechanics_a = np.empty(2)
    system_SET_oscillators_mechanics_v_dot_tmp = np.empty(2)
    system_SET_oscillators_mechanics_v_dot = np.empty(2)
    system_te0_mechanics_k, system_oscillator0_mechanics_v_dot, system_oscillator0_mechanics_x_dot, system_oscillator1_mechanics_v_dot, system_oscillator1_mechanics_x_dot = variables[4:9]
    system_spc3_mechanics_k = system_te0_mechanics_k
    system_spc3_mechanics_x2 = system_SET_oscillators_mechanics_x[1]
    system_spc3_mechanics_x1 = system_SET_oscillators_mechanics_x[0]
    system_spc3_mechanics_c, system_spc3_mechanics_F1, system_spc3_mechanics_F2 = Spring_Equation_eval(system_spc3_mechanics_x1, system_spc3_mechanics_x2, system_spc3_mechanics_k)
    system_SET_couplings_mechanics_x2[0], system_SET_couplings_mechanics_x2[1] = system_SET_oscillators_mechanics_x[1], system_SET_oscillators_mechanics_x[1]
    system_SET_couplings_mechanics_x1[0], system_SET_couplings_mechanics_x1[1] = system_SET_oscillators_mechanics_x[0], system_SET_oscillators_mechanics_x[0]
    for i in range(2):
        system_SET_couplings_mechanics_c[i], system_SET_couplings_mechanics_F1[i], system_SET_couplings_mechanics_F2[i] = Spring_Equation_eval(system_SET_couplings_mechanics_x1[i], system_SET_couplings_mechanics_x2[i],
            system_SET_couplings_mechanics_k[i])
    for i in range(2):
        system_SET_oscillators_mechanics_x_dot[i], system_SET_oscillators_mechanics_a[i], system_SET_oscillators_mechanics_v_dot_tmp[i] = DampenedOscillator_eval(system_SET_oscillators_mechanics_v[i], system_SET_oscillators_mechanics_k[i],
            system_SET_oscillators_mechanics_x[i], system_SET_oscillators_mechanics_c[i])
    system_SET_oscillators_mechanics_v_dot[0], system_SET_oscillators_mechanics_v_dot[1] = system_SET_couplings_mechanics_F1[0] + system_SET_couplings_mechanics_F1[1] + system_spc3_mechanics_F1 + system_SET_couplings_mechanics_F2[0
        ] + system_SET_couplings_mechanics_F2[1] + system_spc3_mechanics_F2, system_SET_couplings_mechanics_F1[0] + system_SET_couplings_mechanics_F1[1] + system_spc3_mechanics_F1 + system_SET_couplings_mechanics_F2[0
        ] + system_SET_couplings_mechanics_F2[1] + system_spc3_mechanics_F2
    system_SET_oscillators_mechanics_v_dot += system_SET_oscillators_mechanics_v_dot_tmp
    variables[11:21] = system_SET_oscillators_mechanics_v_dot[0], system_SET_oscillators_mechanics_x_dot[0], system_SET_oscillators_mechanics_v_dot[1], system_SET_oscillators_mechanics_x_dot[1
        ], system_spc3_mechanics_x1, system_spc3_mechanics_x2, system_spc3_mechanics_F1, system_spc3_mechanics_c, system_spc3_mechanics_F2, system_spc3_mechanics_k
    variables[0:4] = system_SET_oscillators_mechanics_v[0], system_SET_oscillators_mechanics_x[0], system_SET_oscillators_mechanics_v[1], system_SET_oscillators_mechanics_x[1]
    return variables[11:15]
