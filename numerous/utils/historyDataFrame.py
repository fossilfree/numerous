import pandas as pd
from datetime import timedelta
import numpy as np
from numba import int64, float64

from utils.callback_decorators import NumbaCallback, CallbackMethodType
from utils.numba_callback import NumbaCallbackBase



class NumbaCallback(object):

    def __init__(self):
        self.numba_params_spec = {}

    def register_numba_varaible(self, variable_name, numba_variable_type):
        self.numba_params_spec.update({variable_name: numba_variable_type})


class HistoryDataFrame(NumbaCallback):
    """
       Historian class that saves states after each finished step of the simulation.

       Attributes
       ----------
             data :  dataframe
                  Structured array to save data.
             time :  Timestamp
                  either time  from the begging of simulation

            filter : :class:`numerous.engine.utils.OutputFilter`
       """

    def __init__(self, start=None, filter=None, **kwargs):
        super().__init__()
        self.df = pd.DataFrame(columns=['time'])
        self.df = self.df.set_index('time')
        if start:
            self.df.index = pd.to_datetime(self.df.index)
        else:
            self.df.index = pd.to_timedelta(self.df.index, unit='s')

        self.time = start if start else timedelta(0)
        if filter:
            self.filter = filter
        else:
            self.filter = OutputFilter()
        # self.callback = _SimulationCallback("save to dataframe", priority=-1)
        # self.callback.add_callback_function(self.update)
        # self.callback.add_finalize_function(self.finalize)
        self.list_result = []

    def update(self, time, variables, **kwargs):
        if self.filter:
            variables = self.filter.filter_varaibles(variables)

        keys = []
        for key in variables.keys():
            if variables[key].alias:
                keys.append(variables[key].alias)
            else:
                keys.append(key)

        self.list_result.append(pd.DataFrame([[y.get_value() for y in variables.values()]],
                                             columns=keys, index=[self.time +
                                                                  timedelta(seconds=self.time.total_seconds() + time)]))

    def finalize(self):
        self.df = pd.concat([new_row for new_row in self.list_result], ignore_index=False)

    @staticmethod
    def load(filename):
        """
        Parameters
        ----------
        filename:string
            path to file that contains the stored history dataframe

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

    def finalize(self):
        time = self.data[0]
        data = {'time': time}

        for i, var in enumerate(self.var_list):
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
