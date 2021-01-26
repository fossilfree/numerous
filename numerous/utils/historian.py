import os

class Historian:
    def __init__(self, max_size):
        self.max_size = max_size

    def get_historian_max_size(self):
        return self.max_size

    def store(self, df):
        pass

class InMemoryHistorian(Historian):

    def __init__(self):
        super().__init__(None)



class LocalHistorian(Historian):

    def __init__(self, filename, max_size=500):
        super().__init__(max_size)
        self.filename = filename

    def store(self, df):
        if os.path.isfile(self.filename):
            df.dropna().to_csv(self.filename, mode='a', header=False)
        else:
            df.dropna().to_csv(self.filename)
