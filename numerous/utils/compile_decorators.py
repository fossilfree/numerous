from numba import njit, objmode

no_options = False
fastmath = True
parallel = False


def njit_(f):
    if no_options:
        return njit(f)
    return njit(f, fastmath=fastmath, parallel=parallel)


def nojit(f):
    return f


basic_njit = njit_

