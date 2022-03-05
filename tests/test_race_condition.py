import numpy as np
import pytest

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.simulation.solvers.base_solver import solver_types
from numerous.engine.system import Subsystem, Item
from numerous.multiphysics import EquationBase, Equation




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
    def __init__(self, tag='system_race', item=object):
        super().__init__(tag)
        self.register_item(item)


def analytical(tvec, o, a, dt):
    s_delayed = np.zeros(len(tvec))
    s = np.zeros(len(tvec))
    for i, t in enumerate(tvec):
        s_delayed[i] = a / o * (np.cos(o * dt) - np.cos(o * t))
        s[i] = a / o * (1 - np.cos(o * t))

    return s, s_delayed


@pytest.mark.parametrize("use_llvm", [True,False])
def test_race_condition_1(solver, use_llvm):
    omega0 = 0.01
    dt = 10
    s1 = System(item=Link(item1=Item1(omega=omega0)),tag='system_race_1')
    s2 = System(item=Item2(omega=omega0),tag='system_race_2')

    m1 = Model(s1,use_llvm=use_llvm)
    sim1 = Simulation(m1, num=500, max_step=dt)


    sim1.solve()

    df1 = sim1.model.historian_df

    m2 = Model(s2, use_llvm=use_llvm)
    sim2 = Simulation(m2, num=500, max_step=dt)
    sim2.solve()
    df2 = sim2.model.historian_df

    assert np.all(
        np.isclose(np.array(df1['system_race_1.link.t1.S'])[2:], np.array(df2['system_race_2.item2.t1.S'][2:]), rtol=1e-02, atol=1e-04))

