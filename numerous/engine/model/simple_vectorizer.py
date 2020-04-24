from numba import njit
import numpy as np

def simple_vectorize(f):
    f_ = njit(f)
    def simple_vectorized(self,scopes):
        l = np.shape(scopes)[0]
        for i in range(l):
            f_(scopes[i,:])


    return simple_vectorized