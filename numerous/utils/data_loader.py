from abc import ABC
from typing import Tuple

import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas.errors import EmptyDataError


class DataLoader(ABC):

    def __init__(self):
        self.chunksize = 1000

    def load(self, df_id: str, t: float):
        pass


class CSVDataLoader(DataLoader):

    def __init__(self, chunksize=1000):
        super(CSVDataLoader, self).__init__()
        self.chunksize = chunksize
        self.is_chunks = False
        self._skiprows = 0
        self.df = pd.DataFrame()
        self.t_ix = None
        if self.chunksize:
            self.is_chunks = True

    def _valid(self, chunks, t):
        for i, chunk in enumerate(chunks):
            if not self.t_ix:
                self.t_ix = np.where(chunk.columns == 'time')[0][0]
            mask = chunk.iloc[:, self.t_ix] <= t

            if mask.all():
                yield chunk
            else:
                self._skiprows += i * self.chunksize
                yield chunk
                break

    def load(self, df_id: str, t: float) -> DataFrame:
        if not self.df.empty and t < min(self.df.iloc[:, self.t_ix]):
            self._skiprows = 0

        if self.is_chunks:
            chunks = pd.read_csv(df_id, header=0, skiprows=self._skiprows, chunksize=self.chunksize)
            df = pd.concat(self._valid(chunks, t))
        else:
            df = pd.read_csv(df_id, header=0)

        self.df = df
        return df



class InMemoryDataLoader(DataLoader):

    def __init__(self, df):
        super(InMemoryDataLoader, self).__init__()
        self.df = df

    def load(self, df_id: str, t: float) -> DataFrame:
        return self.df

