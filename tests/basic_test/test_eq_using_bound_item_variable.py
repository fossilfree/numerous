import numpy as np
import pytest

from numerous.multiphysics.equation_decorators import Equation, NumerousFunction
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine import model, simulation
from numerous.engine.simulation.solvers.base_solver import solver_types

expected = 17

class ItemRefer(EquationBase, Item):
    def __init__(self, tag="tm"):

        Item.__init__(self, tag)
        EquationBase.__init__(self, tag)

        self.add_parameter('x_1', 0)
        self.add_parameter('x_2', expected)

        mechanics = self.create_namespace('test_nm')
        self.mechanics = mechanics
        mechanics.add_equations([self])


class TestReferToItem(EquationBase, Item):
    def __init__(self, item_refer, tag="tm"):

        Item.__init__(self, tag)
        EquationBase.__init__(self, tag)

        self.item_refer = item_refer
        self.add_parameter('x', 0)

        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x = self.item_refer.mechanics.x_2
        self.item_refer.mechanics.x_1 = scope.x

class IfSystem(Subsystem):
    def __init__(self, tag, *items):
        super().__init__(tag)

        self.register_items(items)


@pytest.mark.parametrize("solver", solver_types[:1])
@pytest.mark.parametrize("use_llvm", [False, True][1:])
def test_external_closure_0(solver, use_llvm):
    item_refer=ItemRefer()
    model_ = model.Model(
        IfSystem('m_system', TestReferToItem(tag='tm1', item_refer=item_refer), item_refer),
        use_llvm=use_llvm)
    s = simulation.Simulation(model_, solver_type=solver, t_start=0, t_stop=3, num=1, num_inner=1)
    s.solve()

    assert s.model.historian_df['m_system.tm1.test_nm.x'][1] == expected
    assert s.model.historian_df['m_system.tm.test_nm.x_1'][1] == expected

