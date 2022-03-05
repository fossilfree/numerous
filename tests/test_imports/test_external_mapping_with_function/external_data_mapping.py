import pytest

from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine import model, simulation
from numerous.engine.simulation.solvers.base_solver import solver_types


class DynamicDataTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(DynamicDataTest, self).__init__(tag)
        self.add_parameter('T1', 5)
        self.add_parameter('T_i1', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T_i1 = h_test(scope.T1)


class DynamicDataSystem(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)
        self.register_items([DynamicDataTest('tm1')])


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_mapping(use_llvm):
    model_ = model.Model(DynamicDataSystem('system'), imports=[("external_data_functions", "h_test")],
                         use_llvm=use_llvm)
    s = simulation.Simulation(model_, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    expected = 106.0
    assert s.model.historian_df['system.tm1.test_nm.T_i1'][1] == expected
