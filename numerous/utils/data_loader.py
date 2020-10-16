from abc import ABC
from typing import Tuple

import pandas as pd
from pandas import DataFrame
from pandas.errors import EmptyDataError


class DataLoader(ABC):

    def __init__(self):
        self.chunksize = 1000

    def load(self, df_id: str, t: int):
        pass


class LocalDataLoader(DataLoader):

    def __init__(self, chunksize=1000):
        self.chunksize = chunksize
        self.is_chunks = False
        if self.chunksize is not None:
            self.is_chunks = True


    def load(self, df_id: str, t: int) -> Tuple[None, bool]:
        if self.is_chunks:
            return pd.read_csv(df_id, header=0, skiprows=t, chunksize=self.chunksize).get_chunk()
        else:
            return pd.read_csv(df_id, header=0, skiprows=t, chunksize=self.chunksize)

