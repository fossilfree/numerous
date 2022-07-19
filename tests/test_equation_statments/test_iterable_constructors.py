import numpy as np
import pytest

from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine import model, simulation


class ListTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        Item.__init__(self, tag)
        EquationBase.__init__(self, tag)
        self.add_parameter('x', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        y = [44, 2, 3]
        scope.x = y[0]


class SetTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        Item.__init__(self, tag)
        EquationBase.__init__(self, tag)
        self.add_parameter('x', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        s = {44, 2, 3}
        scope.x = sorted(list(s))[2]


class System(Subsystem):
    def __init__(self, tag, *items):
        super().__init__(tag)
        self.register_items(items)


@pytest.mark.parametrize("use_llvm", [True, False],)
@pytest.mark.parametrize("test", [ListTest, SetTest, TupleTest])
def test_subscript(use_llvm, test):
    model_ = model.Model(System('m_system', test('tm3')), use_llvm=use_llvm)
    s = simulation.Simulation(model_, t_start=0, t_stop=3, num=1, num_inner=1)
    s.solve()
    expected = 44
    assert s.model.historian_df['m_system.tm3.test_nm.x'][1] == expected

