#!/usr/bin/env python
# coding: utf-8

# In[1]:
from pytest import approx
import pytest
from numerous.engine.system import Subsystem, Item
from numerous.multiphysics import EquationBase, Equation
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np

from numerous.engine.simulation.solvers.base_solver import solver_types

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    import shutil
    shutil.rmtree('./tmp', ignore_errors=True)
    yield

class InitialValue(Item, EquationBase):
    def __init__(self, tag='initialvalue', x0=1):
        super(InitialValue, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_constant('x', x0)
        self.t1.add_equations([self])


class Success1(Subsystem):
    def __init__(self, tag='works_deeplink', N_outer=2, N_inner=5, k=0.9):
        super().__init__(tag)
        inlet_item = InitialValue(x0=1, tag='inlet_0')
        self.outer_counter = 0
        self.inner_counter = 0
        for i in range(N_outer):
            self.outer_counter += 1
            system = Level1(tag='linkersubsystem_deeplink_' + str(i + 1), inlet_item=inlet_item, N=N_inner, k=k)
            self.register_item(system)
            inlet_item = system.ports['outlet']  # Set next inlet item to outlet item from Level1 - this does not work
            # inlet_item = Link(tag='inlet_' +str(i+1), item=system.outlet)
            # self.register_item(inlet_item)


class Success2(Subsystem):
    def __init__(self, tag='doesnotwork_deeplink', N_outer=2, N_inner=5, k=0.9):
        super().__init__(tag)
        inlet_item = InitialValue(x0=1, tag='inlet_0')
        self.outer_counter = 0
        self.inner_counter = 0
        for i in range(N_outer):
            self.outer_counter += 1
            system = Level1(tag='linkersubsystem_deeplink_' + str(i + 1), inlet_item=inlet_item, N=N_inner, k=k)
            self.register_item(system)
            inlet_item = Modify1(tag='inlet_' + str(i + 1), item=system.ports['outlet'])
            self.register_item(inlet_item)


# This subsystem could do some modification of the item object variables
# but for this example it doesnt...
class Modify1(Subsystem, EquationBase):
    def __init__(self, tag='modify1_deeplink', item=object):
        super(Modify1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x', item.t1.x.get_value())
        print(item.t1.x.get_value())
        self.t1.add_equations([self])
        self.register_item(item)

        self.t1.x = item.t1.x


# Yet another subsystem that could do some modification but doesn't
class Modify2(Subsystem, EquationBase):
    def __init__(self, tag='modify2_deeplink', item=object):
        super(Modify2, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x', item.t1.x.get_value())
        print(item.t1.x.get_value())
        self.t1.add_equations([self])
        self.register_item(item)

        self.t1.x = item.t1.x


class Level1(Subsystem):
    def __init__(self, tag='level1_deeplink', N=2, k=0.9, inlet_item=object):
        super().__init__(tag)
        items = []
        inlet_item_next = Modify2(tag='boundary_deeplink', item=inlet_item)
        self.register_item(inlet_item_next)
        for i in range(N):
            item = Base(tag='item_' + str(i + 1), inlet=inlet_item_next, k=k)
            self.register_item(item)
            inlet_item_next = item  # This works

        self.add_port("outlet", item) # Add outlet item as last item in Level1 subsystem


class Base(Subsystem, EquationBase):
    def __init__(self, tag='level2', inlet=object, k=0.9):
        super(Base, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x', 0)
        self.add_parameter('x0', 0)
        self.add_constant('k', k)
        self.t1.add_equations([self])
        self.register_item(inlet)

        # Bind

        self.t1.x0 = inlet.t1.x

    @Equation()
    def eval(self, scope):
        scope.x = scope.x0 * scope.k


def expected(length, N, k):
    return (k ** N) * np.ones(length)

@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_system_link_Success1(solver, use_llvm):
    N_inner = 5
    N_outer = 2
    system = Success1(N_outer=N_outer, N_inner=N_inner)
    model = Model(system,use_llvm=use_llvm)

    sim = Simulation(model, t_start=0, t_stop=100, num=200, solver_type=solver)

    sim.solve()
    df = sim.model.historian_df

    assert approx(np.array(df['works_deeplink.linkersubsystem_deeplink_2.item_5.item_4.item_3.item_2.item_1.boundary_deeplink.t1.x'])[1:], rel=1) == \
           expected(len(df.index[:-1]),  (N_outer-1)*N_inner, 0.9)

@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_system_link_Success2(solver,use_llvm):
    N_inner = 5
    N_outer = 2
    system = Success2(N_outer=N_outer, N_inner=N_inner)
    model = Model(system,use_llvm=use_llvm)

    sim = Simulation(model, t_start=0, t_stop=100, num=200, solver_type=solver)

    sim.solve()
    df = sim.model.historian_df

    assert approx(np.array(df['doesnotwork_deeplink.inlet_2.item_5.item_4.item_3.item_2.item_1.boundary_deeplink.t1.x'])[1:], rel=1) == \
           expected(len(df.index[:-1]), (N_outer-1)*N_inner, 0.9)




