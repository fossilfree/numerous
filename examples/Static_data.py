import os

import pandas

from historian import LocalHistorian, InMemoryHistorian
from numerous.engine.model.external_mappings import ExternalMappingElement, InterpolationType
from numerous.utils.data_loader import LocalDataLoader
from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem

if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    from matplotlib import pyplot as plt


class StaticDataTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(StaticDataTest, self).__init__(tag)

        ##will map to variable with the same path in external dataframe/datasource
        self.add_parameter('T1', 0, )
        self.add_parameter('T2', 0, )
        self.add_parameter('T_i1', 0)
        self.add_parameter('T_i2', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T_i1 = scope.T1
        scope.T_i2 = scope.T2


class StaticDataSystem(Subsystem):
    def __init__(self, tag, n=1):
        super().__init__(tag)
        o_s = []
        for i in range(n):
            o = StaticDataTest('tm' + str(i))
            o_s.append(o)
        # Register the items to the subsystem to make it recognize them.
        self.register_items(o_s)


if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    import pandas as pd
    from matplotlib import pyplot as plt

    external_mappings = []
    malmo_sturup_data_frame = "malmo_sturup-2019.csv"
    index_to_timestep_mapping = 'time'
    index_to_timestep_mapping_start = 0
    time_multiplier = 1
    dataframe_aliases = {
        'system.tm0.test_nm.T1': ("Dew Point Temperature {C}", InterpolationType.PIESEWISE),
        'system.tm0.test_nm.T2': ('Dry Bulb Temperature {C}', InterpolationType.LINEAR)
    }
    external_mappings.append(
        ExternalMappingElement(malmo_sturup_data_frame, index_to_timestep_mapping, index_to_timestep_mapping_start,
                               time_multiplier,
                               dataframe_aliases))
    history_file = "test.csv"
    if os.path.exists(history_file):
        os.remove(history_file)
    histr = LocalHistorian(history_file, 1000)
    # histr = InMemoryHistorian()
    model = model.Model(StaticDataSystem('system', n=1), external_mappings=external_mappings,
                        data_loader=LocalDataLoader(chunksize=None), historian=histr)
    s = simulation.Simulation(model, t_start=0, t_stop=10000.0, num=10000, num_inner=100, max_step=.1)

    # Solve and plot
    tic = time()
    s.solve()
    toc = time()
    historian_df = pandas.read_csv(history_file)
    # historian_df =s.model.historian_df
    print('Execution time: ', toc - tic)
    print(len(list(s.model.historian_df)))
    historian_df['system.tm0.test_nm.T_i1'].plot()
    historian_df['system.tm0.test_nm.T_i2'].plot()
    plt.show()
    plt.interactive(False)
