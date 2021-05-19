import pytest
from pytest import approx

import simulation
from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system import Item, Subsystem
from numerous.engine.model import Model
from numerous.engine.simulation.solvers.base_solver import solver_types

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    import shutil
    shutil.rmtree('./tmp', ignore_errors=True)
    yield


class Item1(Item, EquationBase):
    def __init__(self, tag='item1'):
        super(Item1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x_dot = 1



class System(Subsystem):
    def __init__(self, tag='system'):
        super(System, self).__init__(tag)
        item1 = Item1(tag='item1')
        self.register_items([item1])



@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [False,True])
def test_deriv_order(solver, use_llvm):
    m = System()
    model = Model(m, use_llvm=use_llvm, generate_graph_pdf=True)
    s = simulation.Simulation(model, solver_type=solver, t_start=0, t_stop=2.0, num=2, num_inner=2)
    s.solve()
    historian_df = s.model.historian_df
    assert approx(historian_df['system.item1.t1.x'][2]) == 2

