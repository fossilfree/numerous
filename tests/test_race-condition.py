from pytest import approx

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.utils.historyDataFrame import SimpleHistoryDataFrame
from numerous.engine.system import Subsystem, Item
from numerous.multiphysics import EquationBase, Equation
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt



class Item1(Item, EquationBase):
    def __init__(self, tag='item1', omega=0.5, amplitude=1):
        super(Item1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('t', 0)
        self.add_parameter('P', 0)
        self.add_constant('o', omega)
        self.add_constant('a', amplitude)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.t_dot = 1
        scope.P = scope.a * np.sin(scope.o * scope.t)


class Link(Subsystem, EquationBase):
    def __init__(self, tag='link', item1=object):
        super(Link, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('S', 0)
        self.add_parameter('P', 0)
        self.t1.add_equations([self])
        self.register_item(item1)
        self.t1.P = item1.t1.P

    @Equation()
    def eval(self, scope):
        scope.S_dot = scope.P


class Item2(Item, EquationBase):
    def __init__(self, tag='item2', omega=0.5, amplitude=1):
        super(Item2, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('t', 0)
        self.add_state('S', 0)
        self.add_constant('a', amplitude)
        self.add_constant('o', omega)
        self.add_parameter('P', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.t_dot = 1
        scope.P = scope.a * np.sin(scope.o * scope.t)
        scope.S_dot = scope.P


class System(Subsystem):
    def __init__(self, tag='system', item=object):
        super().__init__(tag)
        self.register_item(item)


def analytical(tvec, o, a, dt):
    s_delayed = np.zeros(len(tvec))
    s = np.zeros(len(tvec))
    for i, t in enumerate(tvec):
        s_delayed[i] = a / o * (np.cos(o * dt) - np.cos(o * t))
        s[i] = a / o * (1 - np.cos(o * t))

    return s, s_delayed




def test_race_condition_1():
    omega0 = 0.01
    dt = 10
    s1 = System(item=Link(item1=Item1(omega=omega0)))
    s2 = System(item=Item2(omega=omega0))

    m1 = Model(s1, historian=SimpleHistoryDataFrame())
    m2 = Model(s2, historian=SimpleHistoryDataFrame())

    sim1 = Simulation(m1, max_step=dt)
    sim2 = Simulation(m2, max_step=dt)

    sim1.solve()

    df1 = sim1.model.historian.df.set_index('time')

    sim2.solve()
    df2 = sim2.model.historian.df.set_index('time')

    f = [df1, df2]
    df = pd.concat(f, axis=1, sort=False)
    assert np.all(np.isclose(np.array(df['system.link.t1.S']), np.array(df['system.item2.t1.S'])))
