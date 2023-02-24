import os
import pandas


class Historian:
    """
    Historian is the base class for storing the simulation results.
    """

    def __init__(self, max_size: int, timesteps_reserved_per_event: int = 100):
        self.max_size = max_size
        self.timesteps_reserved_per_event = timesteps_reserved_per_event

    def get_historian_max_size(self, number_of_timesteps: int, events_count: int) -> int:
        """
          Calculates the maximum size of the historian.

          Parameters:
          number_of_timesteps (int): The number of timesteps expected in the simulation.
          events_count (int): Expected number of events in the simulation.

          Returns:
          int: The maximum size of the historian.
          """
        if not self.max_size:
            ## +1 here since we are using <= to compare inside compiled model
            return number_of_timesteps + events_count * self.timesteps_reserved_per_event + 1
        else:
            return self.max_size

    def store(self, df: pandas.DataFrame):
        """
          Stores the simulation results of the historian.

          Parameters:
          df (pandas.DataFrame): The simulation results to be stored in the historian.
          """
        pass


class InMemoryHistorian(Historian):
    """
    InMemoryHistorian stores simulation results in memory.
    """

    def __init__(self):
        super().__init__(0)


class LocalHistorian(Historian):
    """
    LocalHistorian stores simulation results in a local file when max_size is reached.
    """

    def __init__(self, filename: str, max_size: int):
        """"
        filename (str): The filename of the CSV file to store simulation results.
        max_size (int): The maximum number of rows to store in the dataframe.
        """
        super().__init__(max_size)
        self.filename = filename

    def store(self, df: pandas.DataFrame):
        """

        Stores the simulation results of the historian, into  csv file.

        Parameters:
        df (pandas.DataFrame): The dataframe holding all the simulation results.

        """
        if os.path.isfile(self.filename):
            df.dropna().to_csv(self.filename, mode='a', header=False)
        else:
            df.dropna().to_csv(self.filename)