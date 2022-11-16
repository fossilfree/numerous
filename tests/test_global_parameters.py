import pytest

from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine import model, simulation


class TestGlobalParameter(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(TestGlobalParameter, self).__init__(tag)
        self.add_parameter('T', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T = scope.global_vars.t


class TestGlobalParameterSystem(Subsystem):
    def __init__(self, tag, item):
        super().__init__(tag)
        self.register_items([item])


class TestConstantGlobalParameter(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(TestConstantGlobalParameter, self).__init__(tag)
        self.add_parameter('T', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T = scope.global_vars.constant_value


@pytest.mark.parametrize("use_llvm", [True, False])
def test_time_variable(use_llvm):
    model_ = model.Model(TestGlobalParameterSystem('test_system', TestGlobalParameter('tm1')), use_llvm=use_llvm)
    s = simulation.Simulation(model_, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    import numpy as np
    assert list(s.model.historian_df['test_system.tm1.test_nm.T']) == list(np.arange(11.0))


@pytest.mark.parametrize("use_llvm", [True, False])
def test_constant_global_variable(use_llvm):
    model_ = model.Model(TestGlobalParameterSystem('test_system', TestConstantGlobalParameter('tm1')),
                         use_llvm=use_llvm, global_variables=[("constant_value", 11.0)])

    s = simulation.Simulation(model_, t_start=0, t_stop=10.0, num=10, num_inner=10)
    s.solve()
    assert list(s.model.historian_df['test_system.tm1.test_nm.T'])[-1] == 11.0
