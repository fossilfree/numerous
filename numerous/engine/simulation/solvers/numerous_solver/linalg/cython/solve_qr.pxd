import numpy as np
cimport numpy as np

ctypedef np.float_t DTYPE_t

cdef np.ndarray[DTYPE_t, ndim=1] backward_substitution_old(np.ndarray[DTYPE_t, ndim=2] U, np.ndarray[DTYPE_t, ndim=1] y)


