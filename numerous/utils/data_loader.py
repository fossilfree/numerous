from abc import ABC

import pandas as pd
from pandas import DataFrame


class DataLoader(ABC):

    def __init__(self):
        self.chunksize = 1000

    def load(self, df_id: str, t: int):
        pass


class LocalDataLoader(DataLoader):

    def __init__(self, chunksize=1000):
        self.chunksize = chunksize

    def load(self, df_id: str, t: int) -> DataFrame:
        return pd.read_csv(df_id, header=0, skiprows=t, chunksize=self.chunksize)
