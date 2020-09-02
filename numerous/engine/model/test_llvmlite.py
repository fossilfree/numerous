
from numba import njit
from numba.pycc import CC
import numpy as np
import time
#from examples.dampened_oscillator.global_generated import diff_global
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
        print(dt)
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

    x0 = np.ones(100000)
    #rc=Randomcode(max_lines=lines)
    #rc.generate()
    import examples.dampened_oscillator.global_generated as gg

    test_fun = reload(gg)


    fun = test_fun.diff_global

#    @timeit
#    def funwrap(x):
#        return fun(x)


    #f0=funwrap(x0)



    from diff import diff_global


    tic = time.time()
    l = diff_global(x0)
    toc = time.time()
    print('aot exe time: ', toc - tic)
    print('sum: ',sum(l))
    print('sum: ', sum(x0))

    fun_jit = jitted(fun)
    tic = time.time()
    l = fun_jit(x0)
    toc = time.time()
    print('jit complie time: ', toc - tic)

    tic = time.time()
    l = fun_jit(x0)
    #time.sleep(1)
    toc = time.time()
    print('jit exe time: ', toc - tic)
    print('sum: ', sum(l))




    #print(f'aot error/len: {np.sum(f0-f_aot)}/{len(f_aot)}, jit error/len: {np.sum(f0-f_jit)}/{len(f_jit)}')
    print('\n')



#print (f(x0))