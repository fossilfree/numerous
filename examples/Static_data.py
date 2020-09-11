from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
import numpy as np

if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    from matplotlib import pyplot as plt


class StaticDataTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(StaticDataTest, self).__init__(tag)

        ##will map to variable with the same path in external dataframe/datasource
        self.add_parameter('T', 0, external_mapping=True)
        self.add_parameter('T_i', 0)
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T_i = scope.T


class StaticDataSystem(Subsystem):
    def __init__(self, tag, n=1):
        super().__init__(tag)
        oscillators = []
        for i in range(n):
            # Create oscillator
            oscillator = StaticDataTest('tm' + str(i))
            oscillators.append(oscillator)
        # Register the items to the subsystem to make it recognize them.
        self.register_items(oscillators)


if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    import pandas as pd
    from matplotlib import pyplot as plt

    e_df = pd.DataFrame(np.arange(20).reshape(10, 2),
                        columns=['time', 'system.oscillator0.mechanics.T_i.tm0.mechanics.T'])
    e_df['time'] = np.arange(10)
    e_df.set_index('time', inplace=True)
    # Define simulation
    s = simulation.Simulation(
        model.Model(StaticDataSystem('system.oscillator0.mechanics.T_i', n=1), external_mappings=e_df),
        t_start=0, t_stop=100.0, num=100, num_inner=100, max_step=.1
    )
    # Solve and plot
    tic = time()
    s.solve()
    toc = time()
    print('Execution time: ', toc - tic)
    print(len(list(s.model.historian_df)))
    s.model.historian_df['system.oscillator0.mechanics.T_i.tm0.mechanics.T'].plot()
    plt.show()
    plt.interactive(False)
