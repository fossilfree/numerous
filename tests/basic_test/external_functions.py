from numba import njit


@njit
def if_replacement(x1, x2):
    if x1 > x2:
        x1 = 1
    return x1, 1
