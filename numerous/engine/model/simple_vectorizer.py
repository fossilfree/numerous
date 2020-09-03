from numba import njit, prange
import numpy as np
def simple_vectorize(text):
    _text = text
    def simple_vectorize_inner(f):
        f_ = njit(f)
        f_.text = _text
        def simple_vectorized(self,scopes):
            l = np.shape(scopes)[0]
            for i in prange(l):
                f_(scopes[i,:])
        return simple_vectorized
    return simple_vectorize_inner