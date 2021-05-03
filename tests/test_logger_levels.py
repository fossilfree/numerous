import pytest
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.utils.logger_levels import LoggerLevel
from numerous.multiphysics.equation_base import EquationBase
from numerous.multiphysics.equation_decorators import Equation
from numerous.engine.system.item import Item
from numerous.engine.system.subsystem import Subsystem
from numerous.engine.simulation.solvers.base_solver import solver_types, SolverType
import numpy as np
import copy

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    import shutil
    shutil.rmtree('./tmp', ignore_errors=True)
    yield


# Set logger level fixed on t and p and let s be determined by
class TestLogItem1(Item, EquationBase):
    def __init__(self, tag='testlogitem1'):
        super(TestLogItem1, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('t', 0, logger_level=LoggerLevel.DEBUG)
        self.add_state('s', 1)
        self.add_parameter('p', 1, logger_level=LoggerLevel.INFO)

        self.t1.add_equations([self])
        return

    @Equation()
    def eval(self, scope):
        scope.t_dot = 1
        scope.s_dot = -2*np.exp(scope.t)/((1+np.exp(scope.t))**2)


class TestLogSubsystem1(Subsystem):
    def __init__(self, tag='testlogsubsystem1', item=None):
        super().__init__(tag)
        self.register_item(item)

    def set_logger_levels(self, logger_level):
        for item in self.registered_items:
            item.set_logger_level(logger_level)

def sigmoidlike(t):
    a=np.ones(len(t))
    return a-2*(1/(1+np.exp(-t))-0.5)

@pytest.mark.parametrize("solver", solver_types)
# @pytest.mark.skip(reason="Functionality not implemented in current version")
def test_logger_levels(solver):
    num = 100
    t_stop = 100
    t_start = 0
    tvec = np.linspace(t_start, t_stop, num+1)

    analytical_results = sigmoidlike(tvec)

    prefix = 'testlogsubsystem1.testlogitem1.t1'
    p = f"{prefix}.p"
    t = f"{prefix}.t"
    s = f"{prefix}.s"

    expected_result_1_1 = {t: "not None", p: "not None", s: "not None"}
    expected_result_1_2 = {t: "not None", p: "not None", s: "None"}
    expected_result_1_3 = {t: "None", p: "not None", s: "None"}

    expected_result_2_1 = {t: "not None", p: "not None", s: "not None"}
    expected_result_2_2 = {t: "not None", p: "not None", s: "not None"}
    expected_result_2_3 = {t: "None", p: "not None", s: "None"}

    expected_result_3_1 = {t: "not None", p: "not None", s: "not None"}
    expected_result_3_2 = {t: "not None", p: "not None", s: "not None"}
    expected_result_3_3 = {t: "None", p: "not None", s: "not None"}

    system = TestLogSubsystem1(item=TestLogItem1())

    system.set_logger_levels(LoggerLevel.INFO)


    model = Model(system)
    sim = Simulation(model, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver)

    sim.solve() # "t", "p" and "s" should be logged
    df1_1 = sim.model.historian_df

    model.logger_level = LoggerLevel.DEBUG
    sim.solve() ## "t", "p" and "s" should be logged
    df1_2 = sim.model.historian_df # "p" and ""s" should be logged

    model.logger_level = LoggerLevel.INFO
    sim.solve()
    df1_3 = sim.model.historian_df

    # Increase system logger level

    system.set_logger_levels(LoggerLevel.DEBUG)
    model.logger_level = LoggerLevel.ALL
    sim.solve()
    df2_1 = sim.model.historian_df

    model.logger_level = LoggerLevel.DEBUG
    sim.solve()
    df2_2 = sim.model.historian_df

    model.logger_level = LoggerLevel.INFO
    sim.solve()
    df2_3 = sim.model.historian_df

    # Increase system logger level

    system.set_logger_levels(LoggerLevel.INFO)
    model.logger_level = LoggerLevel.ALL
    sim.solve()
    df3_1 = sim.model.historian_df

    model.logger_level = LoggerLevel.DEBUG
    sim.solve()
    df3_2 = sim.model.historian_df

    model.logger_level = LoggerLevel.INFO #
    sim.solve()
    df3_3 = sim.model.historian_df

    tests = np.zeros((3,3), dtype=object)
    expected_results = np.zeros((3,3), dtype=object)

    tests[0, 0] = df1_1
    tests[0, 1] = df1_2
    tests[0, 2] = df1_3
    tests[1, 0] = df2_1
    tests[1, 1] = df2_2
    tests[1, 2] = df2_3
    tests[2, 0] = df3_1
    tests[2, 1] = df3_2
    tests[2, 2] = df3_3

    expected_results[0, 0] = expected_result_1_1
    expected_results[0, 1] = expected_result_1_2
    expected_results[0, 2] = expected_result_1_3
    expected_results[1, 0] = expected_result_2_1
    expected_results[1, 1] = expected_result_2_2
    expected_results[1, 2] = expected_result_2_3
    expected_results[2, 0] = expected_result_3_1
    expected_results[2, 1] = expected_result_3_2
    expected_results[2, 2] = expected_result_3_3


    for i in range(3):
        for j in range(3):
            df = tests[i, j]
            r = expected_results[i, j]

            test = eval(f"df.get(p) is {r.get(p)}") and eval(f"df.get(t) is {r.get(t)}") and \
                   eval(f"df.get(s) is {r.get(s)}")
            assert test == True, f"test failed for: {i,j}"

            if df.get(s) is not None:
                assert pytest.approx(df[s], abs=1e-5) == analytical_results, f"too large difference between analytical " \
                                                                          f"results and numerical results for: {i,j}"




#run_test(SolverType(0))