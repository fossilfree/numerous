from numerous.declarative import ScopeSpec, ItemsSpec, Module, EquationSpec
from numerous.declarative.variables import Parameter, Constant, State
from numerous.declarative.connector import Connector, get_value_for, set_value_from
from numerous.declarative.generate_system import generate_system
import numpy as np


class DampenedOscillator(Module):
    """
        Equation and item modelling a spring and dampener
    """
    tag = "dampened_oscillator"

    class Mechanics(ScopeSpec):
        # Define variables
        k = Constant(1) # spring constant
        c = Constant(1) # coefficient of friction
        a = Parameter(0) # acceleration
        x, x_dot = State(0) # distance
        v, v_dot = State(0) # velocity

    mechanics = Mechanics()

    coupling = Connector(
        x = set_value_from(mechanics.x),
        F = get_value_for(mechanics.v_dot)
    )

    @EquationSpec(mechanics)
    def eval(self, scope: Mechanics):
        scope.a = -scope.k * scope.x - scope.c * scope.v
        scope.v_dot = scope.a
        scope.x_dot = scope.v


class SpringCoupling(Module):

    tag = "springcoup"

    class Mechanics(ScopeSpec):
        k = Parameter(1)
        c = Parameter(1)
        F1 = Parameter(0)
        F2 = Parameter(0)
        x1 = Parameter(0)
        x2 = Parameter(0)

    mechanics = Mechanics()

    side1 = Connector(
        F = set_value_from(mechanics.F1),
        x = get_value_for(mechanics.x1)
    )

    side2 = Connector(
        F=set_value_from(mechanics.F2),
        x=get_value_for(mechanics.x2)
    )

    def __init__(self, k=1.0):
        super().__init__()

        self.mechanics.set_values(k=k)

    @EquationSpec(mechanics)
    def eval(self, scope: Mechanics):
        scope.c = scope.k

        dx = scope.x1 - scope.x2
        F = np.abs(dx) * scope.c

        scope.F1 = -F if scope.x1 > scope.x2 else F  # [kg/s]

        scope.F2 = -scope.F1


class OscillatorSystem(Module):

    class Items(ItemsSpec):
        # The oscillator system will have two oscillators interacting via the coupling
        oscillator1: DampenedOscillator
        oscillator2: DampenedOscillator
        coupling: SpringCoupling

    items = Items()

    # connect oscillators via the coupling
    items.oscillator1.coupling >> items.coupling.side1
    items.oscillator2.coupling >> items.coupling.side2


    def __init__(self, k=0.01, c=0.001, x0=(1, 2),  tag=None):
        super(OscillatorSystem, self).__init__()

        # Initialized the modules in the oscillator system

        self.items.oscillator1 = DampenedOscillator()
        self.items.oscillator1.mechanics.set_values(k=k, c=c, x=x0[0])

        self.items.oscillator2 = DampenedOscillator()
        self.items.oscillator2.mechanics.set_values(k=k, c=c, x=x0[1])

        self.items.coupling = SpringCoupling(k=k)


if __name__ == "__main__":

    from numerous.engine import simulation
    from numerous.engine import model
    from matplotlib import pyplot as plt
    import logging

    logging.basicConfig(level=logging.DEBUG)

    oscillator_system = OscillatorSystem(k=0.01, c=0.001, x0=[1.0, 2.0])

    system = generate_system('system', oscillator_system)
    # Define simulation
    s = simulation.Simulation(model.Model(system, use_llvm=False), t_start=0, t_stop=500.0, num=1000, num_inner=100,
                              max_step=1)
    # Solve and plot

    s.solve()

    s.model.historian_df[['system.oscillator1.mechanics.x', 'system.oscillator2.mechanics.x']].plot()
    # print()
    plt.show()
    plt.interactive(False)
