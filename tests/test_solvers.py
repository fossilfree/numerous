from pytest import approx

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation

from numerous.engine.system import Subsystem, LoggerLevel
from numerous.multiphysics import EquationBase, Equation
from numerous.engine.simulation.solvers.base_solver import SolverType


class SimpleInt(Subsystem, EquationBase):
    def __init__(self, tag='integrate'):
        super(SimpleInt, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0, logger_level=LoggerLevel.INFO)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x_dot = 1

def test_euler():
    model = Model(SimpleInt(tag='system'))

    n = 10
    sim = Simulation(model=model, t_start=0, t_stop=10, num=n, solver=SolverType.NUMEROUS, method='Euler')
    sim.solve()

    df_1 = sim.model.historian_df


    assert approx(df_1['system.t1.x'], 0.01, 0.01) == df_1['time']

def test_RK45():
    model = Model(SimpleInt(tag='system'))

    n = 10
    sim = Simulation(model=model, t_start=0, t_stop=10, num=n, solver=SolverType.NUMEROUS, method='RK45')
    sim.solve()

    df_1 = sim.model.historian_df


    assert approx(df_1['system.t1.x'], 0.01, 0.01) == df_1['time']