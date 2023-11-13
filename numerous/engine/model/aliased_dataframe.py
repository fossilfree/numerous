from copy import copy
from typing import Sequence

import pandas as pd


class AliasedDataFrame:
    def __init__(self, data, aliases={}, rename_columns=True):
        self.aliases = aliases
        self.rename_columns = rename_columns
        if self.rename_columns:
            tmp = copy(list(data.keys()))
            for key in tmp:
                if key in self.aliases.keys():
                    data[self.aliases[key]] = data.pop(key)
        self.df = pd.DataFrame(data)

    def __getitem__(self, names: str | Sequence[str]):

        if self.rename_columns:
            if isinstance(names, str):
                return self.df[self.aliases[names] if names in self.aliases else names]
            if isinstance(names, Sequence):
                df = self.df[[self.aliases[name] if name in self.aliases else name for name in names]]
                df.columns = list(names)
                return df
        else:
            return self.df[names]

    @property
    def index(self):
        return self.df.index

    def get(self, names: str | Sequence[str]):
        if self.rename_columns:
            if isinstance(names, str):
                return self.df.get(self.aliases[names] if names in self.aliases else names)
            if isinstance(names, Sequence):
                df = self.df.get([self.aliases[name] if name in self.aliases else name for name in names])
                return df
        else:
            return self.df.get(names)
