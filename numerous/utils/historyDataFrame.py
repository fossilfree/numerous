import pandas as pd
from datetime import timedelta

from numerous.utils.output_filter import OutputFilter
from numerous.engine.simulation.simulation_callbacks import _SimulationCallback


class HistoryDataFrame:
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

    def __init__(self, start=None, filter=None):
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
        self.callback = _SimulationCallback("save to dataframe")
        self.callback.add_callback_function(self.update)

    def update(self, time, variables):
        if self.filter:
            variables = self.filter.filter_varaibles(variables)

        keys = []
        for key in variables.keys():
            if variables[key].alias:
                keys.append(variables[key].alias)
            else:
                keys.append(key)

        new_row = pd.DataFrame([[y.value for y in variables.values()]],
                               columns=keys, index=[self.time +
                                                    timedelta(seconds=self.time.total_seconds() + time)])
        self.df = pd.concat([self.df, new_row], ignore_index=False)

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

        self.df.to_csv(filename, index_label='time')

    def get_last_state(self):
        """

        Returns
        -------
            state: dict
                last saved state
        """
        return self.df.tail(1).to_dict()
