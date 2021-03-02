from numba import njit
import numpy as np

array = np.arange(100)+100

@njit
def h_test(q):
    if q>0:
        return (array[int(q)] + 1,1)
    else:
        return 0,1

