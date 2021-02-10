from examples.external_mapping_with_function.external_data_functions import h_test
from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem


if __name__ == "__main__":
    from numerous.engine import model, simulation
    from matplotlib import pyplot as plt

class DynamicDataTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(DynamicDataTest, self).__init__(tag)
        self.add_parameter('T1', 5, )
        self.add_parameter('T_i1', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T_i1 = h_test(scope.T1)

class DynamicDataSystem(Subsystem):
    def __init__(self, tag, n=1):
        super().__init__(tag)
        self.register_items(DynamicDataTest('tm1'))


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
