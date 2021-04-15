import pytest
from pytest import approx

from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system import Item, Subsystem
from numerous.engine.model import Model
from numerous.engine.simulation.solvers.base_solver import solver_types


class Item1(Item, EquationBase):
    def __init__(self, tag='item1'):
        super(Item1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0)
        self.add_state('y', 0)

        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x_dot = 1+0.1*scope.x
        scope.y_dot = 1+0.2*scope.y


class System(Subsystem, EquationBase):
    def __init__(self, tag='system'):
        super(System, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('x_temp', 0)
        self.add_parameter('y_temp', 0)
        self.add_parameter('x_dot_item2', 0)
        self.add_parameter('y_dot_item2', 0)

        self.t1.add_equations([self])

        item1 = Item1(tag='item1')
        item2 = Item1(tag='item2')

        self.t1.x_dot_item2 = item2.t1.x_dot
        self.t1.y_dot_item2 = item2.t1.y_dot

        item1.t1.x_dot += self.t1.x_temp
        item1.t1.y_dot += self.t1.y_temp

        self.register_items([item1, item2])

    @Equation()
    def eval(self, scope):
        scope.x_temp = scope.x_dot_item2 + 1
        scope.y_temp = scope.y_dot_item2 + 0.5


class Main(Subsystem):
    def __init__(self, tag='main'):
        super(Main, self).__init__(tag)
        system = System()
        self.register_item(system)


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [False, True])
def test_deriv_order(solver, use_llvm):
    m = Main()
    model = Model(m, use_llvm=use_llvm)
    import numpy as np
    expected = np.array([2.5, 3., 1., 1.])
    assert approx(model.compiled_compute(np.array([0., 0., 0., 0.]))) == expected
    expected_2 = [3.3, 3.6, 1.6 , 1.4]
    assert approx(model.compiled_compute(np.array([1., 2., 3., 4.]))) == expected_2
