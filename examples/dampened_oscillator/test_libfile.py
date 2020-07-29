from time import time
tic = time()
import examples.dampened_oscillator.libfile as lib
toc = time()

print('compile time: ',toc-tic)

import numpy as np

lib.library(2, np.zeros(10, np.float64), np.array([0, 0], np.int64), np.array([0, 0], np.int64))