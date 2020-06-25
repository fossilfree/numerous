from numerous_solver.compile_decorators import basic_njit as njit
import time
import numpy as np
from scipy import linalg

@njit
def forward_substitution_old(L, b):
    #print(L)
    y = np.zeros(len(b))
    y[0] = b[0] / L[0, 0]
    for i in range(1, len(b)):
        v1 = L[i, 0:i]
        v2 = y[0:i]
        prod = v1 @ v2
        y[i] = (b[i] - prod) / L[i, i]
        #y[i] = (b[i] - np.dot(v1,v2)) / L[i, i]
    return y

#@profile
@njit
def forward_substitution(L, b):
    y = np.zeros(len(b))
    Ldiag = np.diag(L)#.tolist()
    y[0] = b[0] / L[0, 0]

    for i in range(1, len(b)):
        v1 = L[i,:i]
        prod = v1 @ y[:i]
        y[i] = (b[i] - prod) / Ldiag[i]


    return y

@njit
def backward_substitution_old(U, y):
    x = np.zeros(len(y))
    n = len(y) - 1
    x[n] = y[n]/U[n,n]
    for i in range(n-1,-1, -1):
        v1 = U[i, (i+1):(n+1)]
        v2 = x[(i+1):(n+1)]
        prod = v1 @ v2
        x[i] = (y[i] - prod) / U[i, i]

    return x

#@profile
@njit
def backward_substitution(U, y):
    x = np.zeros(len(y))
    n = len(y) - 1
    N = n+1

    U_ravel=U.ravel()
    Udiag = np.diag(U)#U_ravel[diag_idx]
    x[n] = y[n]/Udiag[n]

    for i in range(n-1,-1, -1):
        #v1 = U[i, :]
        v1 = U_ravel[(N+1)*i+1:((i+1)*N)]
        v2 = np.array(x[(i+1):(n+1)])
        prod = v1 @ v2#np.dot(v1, v2)
        x[i] = (y[i] - prod) / Udiag[i]
        #x[i] = (y[i] - np.dot(v1,v2))/ U[i, i]
    return np.array(x)

@njit
def solve_cholesky(a, b):
    L = np.linalg.cholesky(a)#
    #y = linalg.solve_triangular(L, b, lower=True)
    #x = linalg.solve_triangular(L.T, y, lower=False)
    y=forward_substitution(L, b)
    x=backward_substitution(np.ascontiguousarray(L.T), y)
    return x


#@profile
@njit
def solve_triangular(L, b):
    y = forward_substitution(L, b)
    x = backward_substitution_old(np.ascontiguousarray(L.T), y)
    #y = linalg.solve_triangular(L,b, lower=True)
    #x = linalg.solve_triangular(L.T, y, lower=False)

    return x

@njit
def solve_qr(Q, R, b):
    bh = Q.T @ b
    x = backward_substitution_old(R, bh)
    return x

@njit
def solve_LU(L, U, b):
    y = forward_substitution(L, b)
    x = backward_substitution_old(U, y)
    return x


def solve_qr_tot(A, b):
    Q, R = np.linalg.qr(A)
    bh = Q.T @ b
    x = backward_substitution(R, bh)
    return x

def timeit(_func, *args):
    start=time.time()
    val=_func(*args)
    end = time.time()
    dt=end-start
    return val, dt

if __name__ == "__main__":
    # a = np.zeros((N, N))
    N = 1000
    a = np.random.rand(N, N)
    a = np.matmul(a, a.T)

    b = np.random.rand(N)



    """
    print("forward substitution:")

    print("myalgo")
    xf_myalgo = forward_substitution(L, b)
    print(xf_myalgo)
    print("scipy")
    xf_scipy = scipy.linalg.solve_triangular(L, b, lower=True)
    print(xf_scipy)

    print("backwards substitution:")

    print("myalgo")
    xb_myalgo = backward_substitution(L.T.conj(), b)
    print(xb_myalgo)
    print("scipy")
    xb_scipy = scipy.linalg.solve_triangular(L.T, b, lower=False)
    print(xb_scipy)
    
    """
    #solve_triangular(a, b)

    x_scipy,dt_scipy=timeit(np.linalg.solve, a,b)
    x_myalgo,dt_myalgo=timeit(solve_qr_tot, a,b)

    print(np.sum(b-np.dot(a,x_scipy)), dt_scipy)
    print(np.sum(b-np.dot(a,x_myalgo)), dt_myalgo)


