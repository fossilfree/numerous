import ctypes
from ctypes import CDLL
import numpy as np
import os

from numba.extending import get_cython_function_address as gcf

libc = CDLL(os.path.abspath("cython.cpython-37m-x86_64-linux-gnu.so"))
solve_qr = libc.__pyx_f_8solve_qr_solve_qr_fun




#addr = gcf("cython", "cython")

A = np.random.rand(3,3)
b = np.random.rand(3)
a = A @ A.T

Q, R = np.linalg.qr(a)

Q = np.ascontiguousarray(Q)
R = np.ascontiguousarray(R)
b = np.ascontiguousarray(b)

solve_qr(ctypes.c_int(1), ctypes.c_int(2), ctypes.c_int(3))



# https://www.semicolonworld.com/question/59387/how-to-convert-pointer-to-c-array-to-python-array