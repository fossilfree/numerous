from numba import jit
import numpy as np
import time
from numba import guvectorize, vectorize,float64
from numba import int64
from inspect import signature
n=10000000
x = np.arange(10*n).reshape(n, 10)


def go_slow(b):
    a = np.copy(b)
    a[1] = a[2] + 10
    a[3] = a[2] + 10
    a[4] = a[2] + 10
    if (a[2] < a[3]) and (a[5] > a[6]):
        a[5] = a[2] + 10
    a[2] = 11 - a[0]
    return a

@guvectorize(['void(float64[:])'],'(n)',nopython=True)
def eval(scope):
    scope[1] = scope[2] + 10
    scope[3] = scope[5] + 10
    scope[4] = scope[4] + 10
    if (a[2]< a[3]) and (a[5] > a[6]):
        a[5] = a[2] + 10
    a[2] = 11 - a[0]

go_slow = np.vectorize(go_slow,signature='(n)->(n)')
# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
sig = signature(go_slow)
x = np.arange(10*n).reshape(n, 10)
start = time.time()
x = go_slow(x)
end = time.time()
print("Elapsed (no compilation) = %s" % (end - start))
x1 = np.arange(10*n).reshape(n, 10)
# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.time()
go_fast(x1)
# print(x_1-x)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))
x2 = np.arange(10*n).reshape(n, 10)
# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.time()

go_fast(x2)
end = time.time()
# print(x_2-x)
print("Elapsed (after compilation) = %s" % (end - start))