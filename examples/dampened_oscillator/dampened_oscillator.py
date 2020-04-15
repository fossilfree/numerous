from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
import numpy as np


class DampenedOscillator(EquationBase, Item):
    """
        Equation and item modelling a spring and dampener
    """
    def __init__(self, tag="tm", x0=1, k=1, c=1, a=1):
        super(DampenedOscillator, self).__init__(tag)

        #define variables
        self.add_constant('k', k)
        self.add_constant('c', c)
        self.add_constant('a', a)
        self.add_state('x', x0)
        self.add_state('v', 0)

        #define namespace and add equation
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):

        #Implement equations for the dampened oscillation
        scope.v_dot = -scope.k * scope.x - scope.c * scope.v + scope.a*np.sign(scope.v)
        scope.x_dot = scope.v


class OscillatorSystem(Subsystem):
    def __init__(self, tag, c=1, k=1, x0=10, a=1, n=1):
        super().__init__(tag)
        oscillators = []
        for i in range(n):
            #Create oscillator
            oscillator = DampenedOscillator('oscillator'+str(i), k=k, c=c, x0=x0, a=a)
            oscillators.append(oscillator)
        #Register the items to the subsystem to make it recognize them.
        self.register_items(oscillators)

if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    from matplotlib import pyplot as plt

    # Define simulation
    s = simulation.Simulation(
        model.Model(OscillatorSystem('system',  c=0, a=0, n=10000)),
        t_start=0, t_stop=100.0, num=100, num_inner=100, max_step=.1
    )
    # Solve and plot
    tic = time()
    s.solve()
    toc = time()
    print('Execution time: ', toc-tic)
    print(len(list(s.model.historian.df)))
    s.model.historian.df['system.oscillator0.mechanics.x'].plot()
    plt.show()
    plt.interactive(False)