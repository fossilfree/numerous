# import pytest
# from numerous.engine.model import Model
# from numerous.engine.simulation import Simulation
# from numerous.utils.logger_levels import LoggerLevel
# from numerous.multiphysics.equation_base import EquationBase
# from numerous.multiphysics.equation_decorators import Equation
# from numerous.engine.system.item import Item
# from numerous.engine.system.subsystem import Subsystem
# from numerous.engine.simulation.solvers.base_solver import solver_types, SolverType
# import numpy as np
# import copy
#
# @pytest.fixture(autouse=True)
# def run_before_and_after_tests():
#     import shutil
#     shutil.rmtree('./tmp', ignore_errors=True)
#     yield
#
#
#
# class TestLogItem1(Item, EquationBase):
#     def __init__(self, tag='testlogitem1'):
#         super(TestLogItem1, self).__init__(tag)
#         self.t1 = self.create_namespace('t1')
#         self.add_state('t', 0, logger_level=LoggerLevel.DEBUG)
#         self.add_state('s', 1)
#         self.add_parameter('p', 1, logger_level=LoggerLevel.INFO)
#
#         self.t1.add_equations([self])
#         return
#
#     @Equation()
#     def eval(self, scope):
#         scope.t_dot = 1
#         scope.s_dot = -2*np.exp(scope.t)/((1+np.exp(scope.t))**2)
#
#
# class TestLogSubsystem1(Subsystem):
#     def __init__(self, tag='testlogsubsystem1', item=None, logger_level=None):
#         super().__init__(tag)
#         self.register_item(item)
#         if logger_level is not None:
#             item.set_logger_level(logger_level)
#
# def sigmoidlike(t):
#     a=np.ones(len(t))
#     return a-2*(1/(1+np.exp(-t))-0.5)
#
# @pytest.mark.parametrize("solver", solver_types)
# @pytest.mark.skip(reason="Functionality not implemented in current version")
# def test_logger_levels(solver):
#     num = 100
#     t_stop = 100
#     t_start = 0
#     tvec = np.linspace(t_start, t_stop, num+1)
#
#     analytical_results = sigmoidlike(tvec)
#
#     prefix = 'testlogsubsystem1.testlogitem1.t1'
#     p = f"{prefix}.p"
#     t = f"{prefix}.t"
#     s = f"{prefix}.s"
#
#     s1 = TestLogSubsystem1(item=TestLogItem1())
#     s2 = TestLogSubsystem1(item=TestLogItem1(), logger_level=LoggerLevel.DEBUG)
#     s3 = TestLogSubsystem1(item=TestLogItem1(), logger_level=LoggerLevel.INFO)
#
#     m1_1 = Model(s1)
#     m1_2 = Model(copy.copy(s1), logger_level=LoggerLevel.DEBUG) # only values flagged with DEBUG or above will be logged
#     m1_3 = Model(copy.copy(s1), logger_level=LoggerLevel.INFO) # only values flagged with INFO will be logged
#
#     sim1_1 = Simulation(m1_1, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver) # "t", "p" and "s" should be logged
#     sim1_2 = Simulation(m1_2, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver) # "t" and "p" should be logged
#     sim1_3 = Simulation(m1_3, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver) # "p" should be logged
#
#     expected_result_1_1 = {t: "not None", p: "not None", s: "not None"}
#     expected_result_1_2 = {t: "not None", p: "not None", s: "None"}
#     expected_result_1_3 = {t: "None", p: "not None", s: "None"}
#
#     m2_1 = Model(s2)
#     m2_2 = Model(copy.copy(s2), logger_level=LoggerLevel.DEBUG)  # only values flagged with DEBUG or above will be logged
#     m2_3 = Model(copy.copy(s2), logger_level=LoggerLevel.INFO)  # only values flagged with INFO will be logged
#
#     sim2_1 = Simulation(m2_1, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver)  # "t", "p" and "s" should be logged
#     sim2_2 = Simulation(m2_2, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver)  # "t", "p" and "s" should be logged
#     sim2_3 = Simulation(m2_3, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver)  # "p" should be logged
#
#     expected_result_2_1 = {t: "not None", p: "not None", s: "not None"}
#     expected_result_2_2 = {t: "not None", p: "not None", s: "not None"}
#     expected_result_2_3 = {t: "None", p: "not None", s: "None"}
#
#     m3_1 = Model(s3)
#     m3_2 = Model(copy.copy(s3), logger_level=LoggerLevel.DEBUG)  # only values flagged with DEBUG or above will be logged
#     m3_3 = Model(copy.copy(s3), logger_level=LoggerLevel.INFO)  # only values flagged with INFO will be logged
#
#     sim3_1 = Simulation(m3_1, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver)  # "t", "p" and "s" should be logged
#     sim3_2 = Simulation(m3_2, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver)  # "t", "p", and "s" should be logged
#     sim3_3 = Simulation(m3_3, t_start=t_start, t_stop=t_stop, num=num, solver_type=solver)  # "s" "p" should be logged
#
#     expected_result_3_1 = {t: "not None", p: "not None", s: "not None"}
#     expected_result_3_2 = {t: "not None", p: "not None", s: "not None"}
#     expected_result_3_3 = {t: "None", p: "not None", s: "not None"}
#
#     sim1_1.solve()
#     sim1_2.solve()
#     sim1_3.solve()
#     sim2_1.solve()
#     sim2_2.solve()
#     sim2_3.solve()
#     sim3_1.solve()
#     sim3_2.solve()
#     sim3_3.solve()
#
#     df1_1 = sim1_1.model.historian_df
#     df1_2 = sim1_2.model.historian_df
#     df1_3 = sim1_3.model.historian_df
#     df2_1 = sim2_1.model.historian_df
#     df2_2 = sim2_2.model.historian_df
#     df2_3 = sim2_3.model.historian_df
#     df3_1 = sim3_1.model.historian_df
#     df3_2 = sim3_2.model.historian_df
#     df3_3 = sim3_3.model.historian_df
#
#     tests = np.zeros((3,3), dtype=object)
#     expected_results = np.zeros((3,3), dtype=object)
#
#     tests[0, 0] = df1_1
#     tests[0, 1] = df1_2
#     tests[0, 2] = df1_3
#     tests[1, 0] = df2_1
#     tests[1, 1] = df2_2
#     tests[1, 2] = df2_3
#     tests[2, 0] = df3_1
#     tests[2, 1] = df3_2
#     tests[2, 2] = df3_3
#
#     expected_results[0, 0] = expected_result_1_1
#     expected_results[0, 1] = expected_result_1_2
#     expected_results[0, 2] = expected_result_1_3
#     expected_results[1, 0] = expected_result_2_1
#     expected_results[1, 1] = expected_result_2_2
#     expected_results[1, 2] = expected_result_2_3
#     expected_results[2, 0] = expected_result_3_1
#     expected_results[2, 1] = expected_result_3_2
#     expected_results[2, 2] = expected_result_3_3
#
#
#     for i in range(3):
#         for j in range(3):
#             df = tests[i, j]
#             r = expected_results[i, j]
#
#             test = eval(f"df.get(p) is {r.get(p)}") and eval(f"df.get(t) is {r.get(t)}") and \
#                    eval(f"df.get(s) is {r.get(s)}")
#             assert test == True, f"test failed for: {i,j}"
#
#             if df.get(s) is not None:
#                 assert pytest.approx(df[s], abs=1e-5) == analytical_results, f"too large difference between analytical " \
#                                                                           f"results and numerical results for: {i,j}"
#
#
#
#
# #run_test(SolverType(0))