import pytest
from pytest import approx

from numerous import EquationBase, Equation
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.simulation.solvers.base_solver import solver_types
from numerous.engine.system import Subsystem, Item
import numpy as np


@pytest.fixture
def test_eq1():
    class TestEq1(EquationBase):
        def __init__(self, P=10):
            super().__init__(tag='example_1')
            self.add_parameter('P', P)
            self.add_parameter('P2', 0)

        @Equation()
        def eval(self, scope):
            scope.P2 = 11

    return TestEq1()


@pytest.fixture
def simple_item(test_eq1):
    class T1(Item):
        def __init__(self, tag):
            super().__init__(tag)

            t1 = self.create_namespace('t1')

            t1.add_equations([test_eq1])

    return T1('test_item')

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    import shutil
    shutil.rmtree('./tmp', ignore_errors=True)
    yield


@pytest.fixture
def ms4(simple_item):
    class S1(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            self.register_items([simple_item])

    return S1('S1_var')


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_var_not_used(ms4, solver, use_llvm):
    m1 = Model(ms4,use_llvm=use_llvm)
    s1 = Simulation(m1, t_start=0, t_stop=100, num=10, solver_type=solver)
    s1.solve()
    df = s1.model.historian_df
    assert approx(np.array(df['S1_var.test_item.t1.P2'])) == np.repeat(11, (11))
