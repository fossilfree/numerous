import pytest

from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine import model, simulation
from numerous.engine.simulation.solvers.base_solver import solver_types


class IfTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(IfTest, self).__init__(tag)
        self.add_parameter('T1', 5)
        self.add_parameter('T_i1', 0)
        self.add_parameter('T_i2', 0)
        self.add_parameter('T_i3', 0)
        self.add_parameter('T_i4', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T_i1, scope.T_i2 = h_test(scope.T1)
        scope.T_i3, scope.T_i4 = h_test(scope.T1 - 100)


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
        c=10
        if scope.T3 > c:
            scope.T_i2 = scope.T1
            scope.T_i4 = scope.T2 -100

import numpy as np


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
        c=100
        if scope.T3 > c:
            scope.T_i2 = scope.T1
            scope.T_i4 = scope.T2 -100



class IfSystem(Subsystem):
    def __init__(self, tag, item):
        super().__init__(tag)
        self.register_items([item])


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_if_statement(solver, use_llvm):
    model_ = model.Model(IfSystem('if_system', IfTest('tm1')), use_llvm=use_llvm,
                         imports=[("external_import_else_if", "h_test")])
    s = simulation.Simulation(model_, solver_type=solver, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    expected_ti1 = 106
    expected_ti4 = 1
    assert s.model.historian_df['if_system.tm1.test_nm.T_i1'][1] == expected_ti1
    assert s.model.historian_df['if_system.tm1.test_nm.T_i4'][1] == expected_ti4


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_if_statement_if_body_executed(solver, use_llvm):
    model_ = model.Model(IfSystem('if_system1', IfTest2('tm1')), use_llvm=use_llvm)
    s = simulation.Simulation(model_, solver_type=solver, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    expected_ti4 = -95
    assert s.model.historian_df['if_system1.tm1.test_nm.T_i4'][1] == expected_ti4


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_if_statement_if_body_skipped(solver, use_llvm):
    model_ = model.Model(IfSystem('if_system2', IfTest3('tm1')), use_llvm=use_llvm)
    s = simulation.Simulation(model_, solver_type=solver, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    expected_ti4 = 0
    assert s.model.historian_df['if_system2.tm1.test_nm.T_i4'][1] == expected_ti4



