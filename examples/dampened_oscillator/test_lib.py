from time import time
import numpy as np
print('compiling...')
tic = time()
from examples.dampened_oscillator.kernel import kernel
toc = time()

print('Compile time: ', toc-tic)

from numba import njit, float64
N = 100000
@njit('void(float64[:], float64[:])')
def test_kernel(variables, y):
    for i in range(N):
        kernel(variables, y)

var = np.ones(50, np.float64)
y = np.ones(6, np.float64)
tic = time()
test_kernel(y, var)
toc = time()

print(f'Exe time - {N} runs: ', toc-tic, ' average: ',(toc-tic)/N)

