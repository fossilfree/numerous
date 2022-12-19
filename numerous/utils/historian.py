import os

import pandas


class Historian:
    """
    Historian is the base class for storing the simulation results.
    """
    def __init__(self, max_size, timesteps_reserved_per_event=100):
        self.max_size = max_size
        self.timesteps_reserved_per_event = timesteps_reserved_per_event

    def get_historian_max_size(self, number_of_timesteps:int, events_count:int):
        """
          Calculates the maximum size of the historian based on the number of timesteps and events in the simulation.

          Parameters:
          number_of_timesteps (int): The number of timesteps in the simulation.
          events_count (int): The number of events in the simulation.

          Returns:
          int: The maximum size of the historian.
          """
        if not self.max_size:
            ## +1 here since we are using <= to compare inside compiled model
            return number_of_timesteps + events_count * self.timesteps_reserved_per_event + 1
        else:
            return self.max_size

    def store(self, df):
        """
          Stores the simulation results in the historian.

          Parameters:
          df (pandas.DataFrame): The simulation results to be stored in the historian.
          """
        pass



class InMemoryHistorian(Historian):
    """
    InMemoryHistorian stores simulation results in memory.
    """

    def __init__(self):
        super().__init__(None)


class LocalHistorian(Historian):
    """
    LocalHistorian stores simulation results in a local file. It has a filename and max_size parameters.
The store method is used to store the simulation results in a local file. If the file already exists,
 the results are appended to it. Otherwise, a new file is created.
    """

    def __init__(self, filename:str, max_size:int):
        super().__init__(max_size)
        self.filename = filename

    def store(self, df: pandas.DataFrame):
        if os.path.isfile(self.filename):
            df.dropna().to_csv(self.filename, mode='a', header=False)
        else:
            df.dropna().to_csv(self.filename)