import os

import pandas

from examples.external_mapping_with_function.external_data_functions import h_test
from historian import LocalHistorian, InMemoryHistorian
from numerous.engine.model.external_mappings import ExternalMappingElement, InterpolationType
from numerous.utils.data_loader import LocalDataLoader
from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
import numpy as np

if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    from matplotlib import pyplot as plt



class DynamicDataTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(DynamicDataTest, self).__init__(tag)
        self.add_parameter('T1', 5, )
        self.add_parameter('T2', 0, )
        self.add_parameter('T_i1', 0)
        self.add_parameter('T_i2', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T_i1 = h_test(scope.T1)


class DynamicDataSystem(Subsystem):
    def __init__(self, tag, n=1):
        super().__init__(tag)
        o_s = []
        for i in range(n):
            o = DynamicDataTest('tm' + str(i))
            o_s.append(o)
        # Register the items to the subsystem to make it recognize them.
        self.register_items(o_s)


if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    import pandas as pd
    from matplotlib import pyplot as plt

    model = model.Model(DynamicDataSystem('system'),external_functions_source="external_data_functions")
    s = simulation.Simulation(model, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    s.model.historian_df['system.tm0.test_nm.T_i1'].plot()
    plt.show()
    plt.interactive(False)
