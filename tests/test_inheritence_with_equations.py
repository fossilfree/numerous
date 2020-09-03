import pytest
from pytest import approx

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation

from numerous.engine.system import Subsystem, Item
from numerous import EquationBase, Equation
from simulation.solvers.base_solver import solver_types


class Base_eq(Item, EquationBase):
    def __init__(self, tag):
        super(Base_eq, self).__init__(tag)

        self.t1 = self.create_namespace('t1')
        self.add_parameter('P', 2)
        self.add_parameter('T', 5)
        self.add_parameter('H', 10)
        self.t1.add_equations([self])

    @Equation()
    def eval2(self, scope):
        scope.P = 5


class Child_eq(Base_eq):
    def __init__(self, tag):
        super(Child_eq, self).__init__(tag)
        self.add_parameter('L', 5)
        self.t1.add_equations([self])

    @Equation()
    def eval1(self, scope):
        scope.L = 7

    @Equation()
    def eval3(self, scope):
        scope.H = 0


@pytest.fixture
def ms1():
    class S1(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            self.register_items([Base_eq("test1")])
    return S1('S1')

@pytest.fixture
def ms2():
    class S2(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            self.register_items([Child_eq("test1")])
    return S2('S2')


@pytest.fixture
def ms3():
    class S2(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            self.register_items([Child_eq("test1"), Child_eq("test2"), Child_eq("test3")])
    return S2('S3')


@pytest.mark.parametrize("solver", solver_types)
def test_equation_inheritence_1(ms1, solver):
    m1 = Model(ms1)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100, solver_type=solver)
    s1.solve()
    assert m1.historian_df["S1.test1.t1.P"][100] == 5

@pytest.mark.parametrize("solver", solver_types)
def test_equation_inheritence_2(ms2, solver):
    m1 = Model(ms2)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100, solver_type=solver)
    s1.solve()
    assert m1.historian_df["S2.test1.t1.P"][100] == 5
    assert m1.historian_df["S2.test1.t1.L"][100] == 7

def test_equation_inheritence_3(ms3):
    m1 = Model(ms3)
    assert len(m1.compiled_eq) == 1