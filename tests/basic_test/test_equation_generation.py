import pytest

from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine import model, simulation




class EqTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(EqTest, self).__init__(tag)
        self.add_parameter('x', 5)
        self.add_parameter('z', 2)
        self.add_parameter('Y', 5)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])


    @Equation()
    def eval(self, scope):
        scope.Y = scope.x+scope.z


class IfSystem(Subsystem):
    def __init__(self, tag, item1,item2,item3):
        super().__init__(tag)

        self.register_items([item1,item2,item3])


@pytest.mark.parametrize("use_llvm", [False,True])
def test_external_if_statement(use_llvm):
    model_ = model.Model(IfSystem('m_system', EqTest('tm1'),EqTest('tm2'),EqTest('tm3')), use_llvm=use_llvm)
    s = simulation.Simulation(model_, t_start=0, t_stop=1.0, num=1, num_inner=1)
    s.solve()
    expected = 7
    assert s.model.historian_df['m_system.tm1.test_nm.Y'][1] == expected
    assert s.model.historian_df['m_system.tm2.test_nm.Y'][1] == expected
    assert s.model.historian_df['m_system.tm3.test_nm.Y'][1] == expected
