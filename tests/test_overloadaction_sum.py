from numerous.engine.system import Item, Subsystem
from numerous.multiphysics import EquationBase, Equation
from numerous.engine.variables import OverloadAction
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np


class Item1(Item, EquationBase):
    def __init__(self, tag='item1'):
        super(Item1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 1)
        self.add_state('t', 0)
        self.t1.add_equations([self], on_assign_overload=OverloadAction.SUM)

    @Equation()
    def eval(self, scope):
        scope.t_dot = 1
        scope.x_dot = -1 * np.exp(-1 * scope.t)


class Subsystem1(Subsystem, EquationBase):
    def __init__(self, tag='subsystem1', item1=object):
        super(Subsystem1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x_dot_mod', 0)
        self.t1.add_equations([self])
        self.register_items([item1])

        item1.t1.x_dot = self.t1.x_dot_mod

    @Equation()
    def eval(self, scope):
        scope.x_dot_mod = -1


class System(Subsystem, EquationBase):
    def __init__(self, tag='system1', subsystem1=object):
        super(System, self).__init__(tag)

        self.register_items([subsystem1])

def expected_sol(t):
    return -1*(t*np.exp(t) -1)*np.exp(-t)


model = Model(System(subsystem1=Subsystem1(item1=Item1())))
sim = Simulation(model, t_start=0, t_stop=10, num=100)
sol = sim.solve()
df = sim.model.historian.df
df['expected_sol'] = expected_sol(np.linspace(0,10,100))
for key in df.keys():
    print(key)
df.plot(y=['subsystem1.item1.t1.x', 'expected_sol'])
plt.show()