import numpy as np
cimport numpy as np

DTYPE = np.float


cdef np.ndarray[DTYPE_t, ndim=1] backward_substitution_old(np.ndarray[DTYPE_t, ndim=2] U, np.ndarray[DTYPE_t, ndim=1] y):
    cdef np.ndarray[DTYPE_t, ndim=1] x = np.zeros(len(y), dtype=DTYPE)
    cdef int n = len(y) - 1
    cdef int i

    x[n] = y[n]/U[n,n]
    for i in range(n-1,-1, -1):
        v1 = U[i, (i+1):(n+1)]
        v2 = x[(i+1):(n+1)]
        prod = v1 @ v2
        x[i] = (y[i] - prod) / U[i, i]

    return x

cdef np.ndarray[DTYPE_t, ndim=1] forward_substitution(np.ndarray[DTYPE_t, ndim=2] L, np.ndarray[DTYPE_t, ndim=1] b):
    cdef np.ndarray[DTYPE_t, ndim=1] y = np.zeros(len(b), dtype=DTYPE)
    y[0] = b[0] / L[0, 0]

    for i in range(1, len(b)):
        v1 = L[i,:i]
        prod = v1 @ y[:i]
        y[i] = (b[i] - prod) / L[i,i]

    return y

def solve_qr_fun(np.ndarray[DTYPE_t, ndim=2] Q, np.ndarray[DTYPE_t, ndim=2] R, np.ndarray[DTYPE_t, ndim=1] b):
    cdef np.ndarray[DTYPE_t, ndim=1] bh = Q.T @ b
    cdef np.ndarray[DTYPE_t, ndim=1] x = backward_substitution_old(R, bh)
    return x

def solve_triangular(np.ndarray[DTYPE_t, ndim=2] L, np.ndarray[DTYPE_t, ndim=1] b):
    cdef np.ndarray[DTYPE_t, ndim=1] y = forward_substitution(L, b)
    cdef np.ndarray[DTYPE_t, ndim=1] x = backward_substitution_old(L.T, y)
    return x