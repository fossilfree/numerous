import numpy as np
import pytest

from numerous.multiphysics.equation_decorators import Equation, NumerousFunction
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine import model, simulation
from numerous.engine.simulation.solvers.base_solver import solver_types


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    import shutil
    shutil.rmtree('./tmp', ignore_errors=True)
    yield


class SelfTest(EquationBase, Item):
    def __init__(self, tag="tm", offset=0):

        Item.__init__(self, tag)
        EquationBase.__init__(self, tag)

        self.add_parameter('x', 0)
        self.add_state('t', 0)

        data = np.arange(100)

        @NumerousFunction()
        def test_self(t):
            return data[int(t)] + offset

        self.test_self = test_self

        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.t_dot = 1
        scope.x = self.test_self(scope.t)


@NumerousFunction()
def closure_func(x):
    return x ** 2


class ClosureFuncTest(EquationBase, Item):
    def __init__(self, tag="tm"):

        Item.__init__(self, tag)
        EquationBase.__init__(self, tag)

        self.add_parameter('x', 0)

        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x = closure_func(5)


test_closure_var = 44.0


class ClosureVarTest(EquationBase, Item):
    def __init__(self, tag="tm"):

        Item.__init__(self, tag)
        EquationBase.__init__(self, tag)

        self.add_parameter('x', 0)

        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x = test_closure_var


class IfSystem(Subsystem):
    def __init__(self, tag, *items):
        super().__init__(tag)

        self.register_items(items)


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [False, True])
def test_external_closure(solver, use_llvm):
    model_ = model.Model(
        IfSystem('m_system', SelfTest('tm1', 1), SelfTest('tm11', 2), ClosureFuncTest('tm2'), ClosureVarTest('tm3')),
        use_llvm=use_llvm)
    s = simulation.Simulation(model_, solver_type=solver, t_start=0, t_stop=3, num=1, num_inner=1)
    s.solve()
    # expected = 3.0
    # assert s.model.historian_df['m_system.tm1.test_nm.x'][1] == expected

    expected = 4.0
    assert s.model.historian_df['m_system.tm11.test_nm.x'][1] == expected

    expected = 25
    assert s.model.historian_df['m_system.tm2.test_nm.x'][1] == expected

    expected = 44
    assert s.model.historian_df['m_system.tm3.test_nm.x'][1] == expected
