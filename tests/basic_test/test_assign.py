import pytest
from pytest import approx

from numerous.engine import simulation
from numerous.engine.model import Model
from numerous.engine.simulation.solvers.base_solver import solver_types
from numerous.engine.system import Item, Subsystem
from numerous.multiphysics.equation_base import EquationBase
from numerous.multiphysics.equation_decorators import Equation


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    import shutil
    shutil.rmtree('./tmp', ignore_errors=True)
    yield


class SingleAssign(Item, EquationBase):
    def __init__(self, tag='item1'):
        super(SingleAssign, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x_dot = 1


class TupleToTupleAssign(Item, EquationBase):
    def __init__(self, tag='item2'):
        super(TupleToTupleAssign, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        el1, el2 = 1, 2
        scope.x_dot = el1


class TupleToVarAssign(Item, EquationBase):
    def __init__(self, tag='item3'):
        super(TupleToVarAssign, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        el1 = 2, 1
        el2, el3 = el1
        scope.x_dot = el3


class TupleToFuncAssign(Item, EquationBase):
    def __init__(self, tag='item4'):
        super(TupleToFuncAssign, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        el1, el3 = if_replacement(100, 50)
        scope.x_dot = el1


class SingleBinaryOperator(Item, EquationBase):
    def __init__(self, tag='item1'):
        super(SingleBinaryOperator, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0)
        self.add_parameter('t', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        res = 2 + scope.t
        scope.x_dot = res - 1


class SingleUnaryOperator(Item, EquationBase):
    def __init__(self, tag='item1'):
        super(SingleUnaryOperator, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0)
        self.add_parameter('t', 2)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        res = -scope.t
        scope.x_dot = res + 3


class System(Subsystem):
    def __init__(self, tag='system', item=None):
        super(System, self).__init__(tag)
        self.register_items([item])


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_single_assign(solver, use_llvm):
    m = System(tag='system1', item=SingleAssign(tag='item1'))
    model = Model(m, use_llvm=use_llvm, generate_graph_pdf=False)
    s = simulation.Simulation(model, solver_type=solver, t_start=0, t_stop=2.0, num=2, num_inner=2)
    s.solve()
    historian_df = s.model.historian_df
    assert approx(historian_df['system1.item1.t1.x'][2]) == 2


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_tuple_assign(solver, use_llvm):
    m = System(tag='system2', item=TupleToTupleAssign(tag='item2'))
    model = Model(m, use_llvm=use_llvm, generate_graph_pdf=False)
    s = simulation.Simulation(model, solver_type=solver, t_start=0, t_stop=2.0, num=2, num_inner=2)
    s.solve()
    historian_df = s.model.historian_df
    assert approx(historian_df['system2.item2.t1.x'][2]) == 2


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.skip(reason="Functionality not implemented in current version")
@pytest.mark.parametrize("use_llvm", [True, False])
def test_tuple_to_var_assign(solver, use_llvm):
    m = System(tag='system3', item=TupleToVarAssign(tag='item3'))
    model = Model(m, use_llvm=use_llvm, generate_graph_pdf=False)
    s = simulation.Simulation(model, solver_type=solver, t_start=0, t_stop=2.0, num=2, num_inner=2)
    s.solve()
    historian_df = s.model.historian_df
    assert approx(historian_df['system3.item3.t1.x'][2]) == 2


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_tuple_to_func_assign(solver, use_llvm):
    m = System(tag='system4', item=TupleToFuncAssign(tag='item4'))
    model = Model(m, use_llvm=use_llvm, generate_graph_pdf=False, imports=[("external_functions", "if_replacement")])
    s = simulation.Simulation(model, solver_type=solver, t_start=0, t_stop=2.0, num=2, num_inner=2)
    s.solve()
    historian_df = s.model.historian_df
    assert approx(historian_df['system4.item4.t1.x'][2]) == 2


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_single_binary_operator(solver, use_llvm):
    m = System(tag='system5', item=SingleBinaryOperator(tag='item5'))
    model = Model(m, use_llvm=use_llvm, generate_graph_pdf=False)
    s = simulation.Simulation(model, solver_type=solver, t_start=0, t_stop=2.0, num=2, num_inner=2)
    s.solve()
    historian_df = s.model.historian_df
    assert approx(historian_df['system5.item5.t1.x'][2]) == 2


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_single_unary_operator(solver, use_llvm):
    m = System(tag='system6', item=SingleUnaryOperator(tag='item6'))
    model = Model(m, use_llvm=use_llvm, generate_graph_pdf=False)
    s = simulation.Simulation(model, solver_type=solver, t_start=0, t_stop=2.0, num=2, num_inner=2)
    s.solve()
    historian_df = s.model.historian_df
    assert approx(historian_df['system6.item6.t1.x'][2]) == 2
