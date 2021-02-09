from numba import njit
import numpy as np

array = np.arange(100)+100

@njit
def h_test(q):
    return array[int(q)] + 1
