from numba import njit, float64
import numpy as np
@njit('void(float64[:])')
def diff_v_dot0(l):
    l[1] += l[0]
    l[3] += l[2]
    l[6] = (np.abs(l[3] - l[1]) - l[4]) * l[5]
    l[7] = l[6]
    l[8] += l[7]
    l[8] = -l[9] * l[2] - l[10] * l[11] + l[12] * np.sign(l[11])


@njit('void(float64[:])')
def diff_x_dot1(l):
    l[1] = l[0]


@njit('void(float64[:])')
def diff_v2_dot2(l):
    l[1] = l[0]


@njit('void(float64[:])')
def diff_v_dot3(l):
    l[1] += l[0]
    l[6] = -l[2] * l[0] - l[3] * l[4] + l[5] * np.sign(l[4])
    l[8] += l[7]
    l[11] = (np.abs(l[8] - l[1]) - l[9]) * l[10]
    l[12] = l[11]
    l[6] += l[12]


@njit('void(float64[:])')
def constant_(l):
    l[1] = l[0] + l[0] * l[0]
    l[3] = float64(2) * l[2]
    l[5] = float64(2) * l[4]


@njit('void(int64, float64[:])')
def switchboard(index, locals):
    if index == 0:
        diff_v_dot0(locals)
    elif index == 1:
        diff_x_dot1(locals)
    elif index == 2:
        diff_v2_dot2(locals)
    elif index == 3:
        diff_v_dot3(locals)
    elif index == 4:
        constant_(locals)
    else:
        raise IndexError('Index out of bounds')

def diff_kernel(variables):


