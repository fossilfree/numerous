
from numba import njit
from numba.pycc import CC
import numpy as np
import time
from randomcode import Randomcode
import os
from importlib import reload



def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)


def kill_numba_cache():

    root_folder = os.path.realpath(__file__ + "/../")

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except Exception as e:
                    print("failed on %s", root)



def timeit(func):
    def timed(*args, **kw):
        start=time.time()
        result = func(*args, **kw)
        end = time.time()
        dt = end-start
        print(f'{func.__name__}: {dt}')
        return result

    return timed

lines_vec = [100]

def jitted(func):
    return njit(func, parallel=False, cache=False)

def compile_aot(func):
    @cc.export('callf', 'f8[:](f8[:])')
    def wrap(func):
        return func
    cc.compile()


for lines in lines_vec:
    kill_numba_cache()
    print(f'lines: {lines}')

    cc = CC('aot_fun')
    valid = False
    while not valid:
        x0 = np.zeros(lines+1)
        rc=Randomcode(max_lines=lines)
        rc.generate()
        import test_fun

        test_fun = reload(test_fun)
        fun = test_fun.fun

        @timeit
        def funwrap(x):
            return fun(x)

        try:
            f0=funwrap(x0)

            valid=True
        except Exception as e:
            pass


    @timeit
    def aotwrap(x):
        compile_aot(fun)
        import aot_fun
        reload(aot_fun)
        callf = aot_fun.callf
        return callf(x)


    x1 = np.zeros(lines + 1)
    f_aot=aotwrap(x1)

    fun_jit = jitted(fun)

    @timeit
    def jitwrap(x):
        return fun_jit(x)


    x2 = np.zeros(lines + 1)
    f_jit=jitwrap(x2)

    print(f'aot error/len: {np.sum(f0-f_aot)}/{len(f_aot)}, jit error/len: {np.sum(f0-f_jit)}/{len(f_jit)}')
    print('\n')



#print (f(x0))