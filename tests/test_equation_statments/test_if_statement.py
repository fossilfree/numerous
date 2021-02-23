import pytest

from test_imports.test_external_mapping_with_function.external_data_functions import h_test
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
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        Q = 2, 3
        scope.T_i1,scope.T_i2 = Q


class IfTest2(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(IfTest2, self).__init__(tag)
        self.add_parameter('T1', 5)
        self.add_parameter('T2', 10)
        self.add_parameter('T_i1', 0)
        self.add_parameter('T_i2', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        if scope.T1 > 4:
            scope.T_i1 = scope.T1
            scope.T_i2 = scope.T2
        else:
            scope.T_i1 = 2
            scope.T_i2 = 3


class IfTest3(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(IfTest3, self).__init__(tag)
        self.add_parameter('T1', 5)
        self.add_parameter('T2', 10)
        self.add_parameter('T_i1', 0)
        self.add_parameter('T_i2', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        if scope.T1 > 10:
            scope.T_i1 = scope.T1
            scope.T_i2 = scope.T2
        else:
            scope.T_i1 = 2
            scope.T_i2 = 3


class IfSystem(Subsystem):
    def __init__(self, tag, item):
        super().__init__(tag)
        self.register_items([item])


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [False])
def test_single_if_statement(solver, use_llvm):
    model_ = model.Model(IfSystem('if_system', IfTest('tm1')), use_llvm=use_llvm)
    s = simulation.Simulation(model_, solver_type=solver, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    expected = 5
    assert s.model.historian_df['if_system.tm1.test_nm.T_i1'][1] == expected

#
# @pytest.mark.parametrize("solver", solver_types)
# @pytest.mark.parametrize("use_llvm", [True, False])
# def test_multiple_line_if_statement(solver, use_llvm):
#     model_ = model.Model(IfSystem('if_system2', IfTest2('tm1')), use_llvm=use_llvm)
#     s = simulation.Simulation(model_, solver_type=solver, t_start=0, t_stop=10.0, num=10, num_inner=10)
#     s.solve()
#     expected_i1 = 5
#     expected_i2 = 10
#     assert s.model.historian_df['if_system2.tm1.test_nm.T_i1'][1] == expected_i1
#     assert s.model.historian_df['if_system2.tm1.test_nm.T_i2'][1] == expected_i2
#
#
# @pytest.mark.parametrize("solver", solver_types)
# @pytest.mark.parametrize("use_llvm", [True, False])
# def test_multiple_line_if_statement2(solver, use_llvm):
#     model_ = model.Model(IfSystem('if_system3', IfTest3('tm1')), use_llvm=use_llvm)
#     s = simulation.Simulation(model_, solver_type=solver, t_start=0, t_stop=10.0, num=10, num_inner=10)
#     s.solve()
#     expected_i1 = 2
#     expected_i2 = 3
#     assert s.model.historian_df['if_system3.tm1.test_nm.T_i1'][1] == expected_i1
#     assert s.model.historian_df['if_system3.tm1.test_nm.T_i2'][1] == expected_i2
