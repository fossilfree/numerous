# import gc
# import sys
# sys.path.append('/home/artem/Source/numerous_github/numerousv2/numerous')
# def trace(frame, event, arg):
#     print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
#     return trace
import numpy as np
from matplotlib import pyplot as plt

from numerous.multiphysics import EquationBase, Equation
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Item, Subsystem
from numerous.engine.system.fmu_subsystem import FMU_Subsystem


class Test_Eq(EquationBase):
    __test__ = False

    def __init__(self, T=0, R=1):
        super().__init__(tag='T_eq')
        self.add_state('Q', T)
        self.add_parameter('R', R)

    @Equation()
    def eval(self, scope):
        scope.Q_dot = scope.R + 9


class G(Item):
    def __init__(self, tag, TG, RG):
        super().__init__(tag)
        t1 = self.create_namespace('t1')
        t1.add_equations([Test_Eq(T=TG, R=RG)])


class S3(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)
        fmu_filename = '/home/artem/fmus/bouncingBall.fmu'
        fmu_subsystem = FMU_Subsystem(fmu_filename, "VanDerPol", debug_output=True)
        self.register_items([fmu_subsystem])

# sys.settrace(trace)
# gc.collect()
subsystem1 = S3('q1')

m1 = Model(subsystem1, use_llvm=False)

s = Simulation(
    m1, t_start=0, t_stop=1, num=10, num_inner=1, max_step=.1)

s.solve()
# gc.set_debug(gc.DEBUG_LEAK)

# fig, ax = plt.subplots()
# # y = np.array(m1.historian_df["q1.Rectifier.t1.outputs"])
# y1 = np.array(m1.historian_df["q1.VanDerPol.t1.h"])
# t = np.array(m1.historian_df["time"])
# ax.plot(t, y1)
# # ax.plot(t, y)


# ax.set(xlabel='time (s)', ylabel='outputs', title='Something')
# ax.grid()
#
# plt.show()

print("execution finished")
