import numpy as np
import pytest

from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine import model, simulation


class SubsciptTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        Item.__init__(self, tag)
        EquationBase.__init__(self, tag)
        self.add_parameter('x', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        y = np.arange(44, 98)
        scope.x = y[int(0)]


class SubscriptSystem(Subsystem):
    def __init__(self, tag, *items):
        super().__init__(tag)
        self.register_items(items)


@pytest.mark.parametrize("use_llvm", [True, False])
def test_subscript(use_llvm):
    model_ = model.Model(SubscriptSystem('m_system', SubsciptTest('tm3')), use_llvm=True)
    s = simulation.Simulation(model_, t_start=0, t_stop=3, num=1, num_inner=1)
    s.solve()
    expected = 44
    assert s.model.historian_df['m_system.tm3.test_nm.x'][1] == expected
