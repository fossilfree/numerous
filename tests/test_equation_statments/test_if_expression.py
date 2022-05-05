import pytest

from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine import model, simulation



class IfTest2(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(IfTest2, self).__init__(tag)
        self.add_parameter('T1', 5)
        self.add_parameter('T2', 5)
        self.add_parameter('T3', 50)
        self.add_parameter('T_i1', 0)
        self.add_parameter('T_i2', 0)
        self.add_parameter('T_i3', 0)
        self.add_parameter('T_i4', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        c = 10
        scope.T_i4 = (scope.T2 - 100) if scope.T3 > c else 1


class IfTest3(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(IfTest3, self).__init__(tag)
        self.add_parameter('T1', 5)
        self.add_parameter('T2', 5)
        self.add_parameter('T3', 50)
        self.add_parameter('T_i1', 0)
        self.add_parameter('T_i2', 0)
        self.add_parameter('T_i3', 0)
        self.add_parameter('T_i4', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        c = 100
        scope.T_i4 = (scope.T2 - 100) if scope.T3 > c else 1


class IfTest4(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(IfTest4, self).__init__(tag)
        self.add_parameter('T1', 5)
        self.add_parameter('T2', 5)
        self.add_parameter('T3', 50)
        self.add_parameter('T_i1', 0)
        self.add_parameter('T_i2', 0)
        self.add_parameter('T_i3', 0)
        self.add_parameter('T_i4', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        c = 100
        q = scope.T3 > c
        scope.T_i4 = (scope.T2 - 100) if q else 1


class IfSystem(Subsystem):
    def __init__(self, tag, item):
        super().__init__(tag)
        self.register_items([item])


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_if_statement_if_body_executed(use_llvm):
    model_ = model.Model(IfSystem('if_system1', IfTest2('tm1')), use_llvm=use_llvm)
    s = simulation.Simulation(model_, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    expected_ti4 = -95
    assert s.model.historian_df['if_system1.tm1.test_nm.T_i4'][1] == expected_ti4


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_if_statement_if_else_executed(use_llvm):
    model_ = model.Model(IfSystem('if_system2', IfTest3('tm1')), use_llvm=use_llvm)
    s = simulation.Simulation(model_, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    expected_ti4 = 1
    assert s.model.historian_df['if_system2.tm1.test_nm.T_i4'][1] == expected_ti4


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_if_statement_if_else_executed_additional_var(use_llvm):
    model_ = model.Model(IfSystem('if_system3', IfTest4('tm1')), use_llvm=use_llvm)
    s = simulation.Simulation(model_, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    expected_ti4 = 1
    assert s.model.historian_df['if_system3.tm1.test_nm.T_i4'][1] == expected_ti4
