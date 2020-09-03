import numba.extending as nbe
import ctypes
import numpy as np
from numba import njit

dpotrf_addr = nbe.get_cython_function_address('scipy.linalg.cython_lapack', 'dpotrf') # cholesky decomposition
dtrsv_addr = nbe.get_cython_function_address('scipy.linalg.cython_blas', 'dtrsv') # solve Ax=b

dpotrf_cfunc = ctypes.CFUNCTYPE(ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p)

dtrsv_cfunc = ctypes.CFUNCTYPE(ctypes.c_int,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p,
                                ctypes.c_void_p)

dpotrf_fun = dpotrf_cfunc(dpotrf_addr)
dtrsv_fun = dtrsv_cfunc(dtrsv_addr)

@njit
def lapack_solve_triangular(Lalloc, balloc, N):
    #solves the linalg equation A*x=L*L**T*x=b, where L is the lower cholesky decomposed matrix of A

    side_a = np.empty(1, dtype=np.int32)
    side_a[0] = 76  # L
    t_a = np.empty(1, dtype=np.int32)
    t_a[0] = 78 # N - solve L*x=b and not L**T*x=b
    diag_a = np.empty(1, dtype=np.int32)
    diag_a[0] = 78
    N_a = np.empty(1, dtype=np.int32)
    N_a[0] = N
    lda_a = np.empty(1, dtype=np.int32)
    lda_a[0] = N
    incx_a = np.empty(1, dtype=np.int32)
    incx_a[0] = 1

    # forward substitution using Lower diagonal matrix L*y = b, solves for y
    dtrsv_fun(side_a.ctypes,
              t_a.ctypes,
              diag_a.ctypes,
              N_a.ctypes,
              Lalloc.ctypes,
              lda_a.ctypes,
              balloc.ctypes,
              incx_a.ctypes
              )

    #side_a[0] = 85 # U
    t_a[0] = 84 # T - since the matrix L**T is the upper matrix
    #t_a[0] = 78 # T

    # backward substitution using upper diagonal matrix U*x=y, solves for x
    dtrsv_fun(side_a.ctypes,
              t_a.ctypes,
              diag_a.ctypes,
              N_a.ctypes,
              Lalloc.ctypes,
              lda_a.ctypes,
              balloc.ctypes,
              incx_a.ctypes
              )

    return balloc


@njit
def lapack_cholesky(side, N, xalloc):

    N_a = np.empty(1, dtype=np.int32)
    N_a[0] = N
    side_a = np.empty(1, dtype=np.int32)
    side_a[0] = side
    z_a = np.empty(1, dtype=np.int32)
    z_a[0] = 1

    dpotrf_fun(side_a.ctypes, N_a.ctypes, xalloc.ctypes, N_a.ctypes, z_a.ctypes)

    # return xalloc, containing the Lower cholesky decomposed matrix of xalloc stored in fortran order.
    # To get the actual lower matrix, transpose xalloc, but know that this slows down all subsequent code for some
    # reason

    return xalloc