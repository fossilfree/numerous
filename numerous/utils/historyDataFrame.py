import pandas as pd
from datetime import timedelta
import numpy as np
from numba import int64, float64

from engine.simulation import Simulation
from utils.callback_decorators import NumbaCallback, CallbackMethodType
from utils.numba_callback import NumbaCallbackBase


class HistoryDataFrameCallback(NumbaCallbackBase):

    def __init__(self):
        self.register_numba_varaible('data', float64[:, :])
        self.register_numba_varaible('ix', int64)

    @NumbaCallback(method_type=CallbackMethodType.INITIALIZE)
    def initialize(self, var_count, number_of_timesteps):
        self.ix = 0
        ##* simulation.num_inner
        self.data = np.empty((var_count + 1, number_of_timesteps), dtype=np.float64)
        self.data.fill(np.nan)

    @NumbaCallback(method_type=CallbackMethodType.UPDATE, run_after_init=True)
    def update(self, time, variables):
        varix = 1
        ix = self.ix
        self.data[0][ix] = time
        for var in variables:
            self.data[varix][ix] = variables[var]
            varix += 1
        self.ix += 1

    def finalize(self):
        time = self.data[0]
        data = {'time': time}

        for i, var in enumerate(self.var_list):
            data.update({var: self.data[i + 1]})

        self.df = pd.DataFrame(data)
        self.df = self.df.dropna(subset=['time'])
        self.df = self.df.set_index('time')
        self.df.index = pd.to_timedelta(self.df.index, unit='s')






