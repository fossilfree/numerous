from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem, ConnectorTwoWay, ItemsStructure

import numpy as np

if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    from matplotlib import pyplot as plt


class Simple(EquationBase, Item):
    """
        Equation and item modelling a spring and dampener
    """

    def __init__(self, tag="simple", x0=1, k=1):
        super(Simple, self).__init__(tag)

        # define variables
        self.add_constant('k', k)

        self.add_state('x', x0)


        # define namespace and add equation
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):

        scope.x_dot = scope.k


class SimpleSystem(Subsystem):
    def __init__(self, tag, k=1, n=1, x0=[10, 8]):
        super().__init__(tag)
        simples = []
        for i in range(n):
            # Create oscillator
            simple = Simple('simple' + str(i), k=k, x0=x0[i])
            simples.append(simple)

        self.register_items(simples, tag="simples", structure=ItemsStructure.SET)



if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    from matplotlib import pyplot as plt

    subsystem = SimpleSystem('system', k=0.01, n=2, x0=[1, 2, 3])
    # Define simulation
    s = simulation.Simulation(
        model.Model(subsystem),
        t_start=0, t_stop=500.0, num=1000, num_inner=100, max_step=1
    )
    # Solve and plot
    tic = time()
    s.solve()
    toc = time()
    print('Execution time: ', toc - tic)
    # print(s.model.historian_df)
    # print(len(list(s.model.historian_df)))
    # s.model.historian_df['oscillator0_mechanics_a'].plot()
    # for i in range(10):
    #    for k, v in zip(list(s.model.historian_df),s.model.historian_df.loc[i,:]):
    #        print(k,': ',v)

    # print(s.model.historian_df.describe())
    # print(list(s.model.historian_df))
    # for c in list(s.model.historian_df):
    #    if not c == 'time':
    # print(s.model.historian_df[c].describe())
    # print(list(s.model.historian_df))

    s.model.historian_df[['system.SET_oscillators.oscillator0.mechanics.x', 'system.SET_oscillators.oscillator1.mechanics.x']].plot()
    # print()
    plt.show()
    plt.interactive(False)
