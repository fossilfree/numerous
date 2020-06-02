import pandas as pd
from datetime import timedelta
import numpy as np
from numba import int64, float64

from numerous.utils.callback_decorators import CallbackMethodType, NumbaCallback
from numerous.utils.numba_callback import NumbaCallbackBase



class NumbaCallback(object):

    def __init__(self):
        super(HistoryDataFrameCallback, self).__init__()
        self.register_numba_varaible('data', float64[:, :])
        self.register_numba_varaible('ix', int64)

        Returns
        -------
        historydataframe: :class:`numerous.engine.utils.HistoryDataFrame`
        """

        df = pd.read_csv(filename)
        hdf = HistoryDataFrame(start=df.index[0])
        hdf.df = df
        return hdf

    def save(self, filename):
        """
        Saves dataframe to the file

        Parameters
        ----------
        filename:
            path to file
        """
        self.finalize()
        self.df.to_csv(filename, index_label='time')

    def get_last_state(self):
        """

        Returns
        -------
            state: dict
                last saved state
        """
        return self.df.tail(1).to_dict()


class SimpleHistoryDataFrame(HistoryDataFrame):
    def __init__(self, start=None, filter=None):
        super().__init__(start, filter)
        # self.callback.add_initialize_function(self.initialize)
        self.data = None
        self.ix = 0
        self.var_list = None
        self.register_numba_varaible('data', float64[:, :])
        self.register_numba_varaible('ix', int64)

    @Equation()
    def update(self, time, variables):
        varix = 1
        ix = self.ix
        self.data[0][ix] = time
        for var in variables:
            self.data[varix][ix] = variables[var]
            varix += 1
        self.ix += 1

    def finalize(self,var_list):
        time = self.data[0]
        data = {'time': time}

        for i, var in enumerate(var_list):
            data.update({var: self.data[i + 1]})

        self.df = pd.DataFrame(data)
        self.df = self.df.dropna(subset=['time'])
        self.df = self.df.set_index('time')
        self.df.index = pd.to_timedelta(self.df.index, unit='s')

    @Equation()
    def numba_initialize(self,var_count,number_of_timesteps):
        self.ix = 0
        self.data = np.empty((var_count + 1, number_of_timesteps), dtype = np.float64)  ##* simulation.num_inner
        # self.data.fill(np.nan)

    def initialize(self, simulation=object):
        variables = simulation.model.path_variables
        var_list = []
        for var in variables:
            var_list.append(var)
        self.var_list = var_list
        self.data = np.ndarray([len(var_list) + 1, len(simulation.time)])  ##* simulation.num_inner
        self.data.fill(np.nan)

        self.update(simulation.time[0], variables)
