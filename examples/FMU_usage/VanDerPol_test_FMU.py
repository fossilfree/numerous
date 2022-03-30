import numpy as np
from matplotlib import pyplot as plt

from numerous import EquationBase, Equation
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation, SolverType
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
        # fmu_filename = '/home/artem/Source/FMPy/Rectifier.fmu'
        fmu_filename = 'VanDerPol.fmu'
        fmu_subsystem = FMU_Subsystem(fmu_filename, "Rectifier", debug_output=True)
        self.register_items([fmu_subsystem])


subsystem1 = S3('q1')
m1 = Model(subsystem1, use_llvm=False)
s = Simulation(
    m1, t_start=0, t_stop=1, num=200, num_inner=200, max_step=.1, solver_type=SolverType.SOLVER_IVP)
# sub_S = m1.system.get_item(ItemPath("q1.BouncingBall"))
s.solve(run_fmu_event_action=True)
# sub_S.fmu.terminate()

fig, ax = plt.subplots()
# t = np.linspace(0, 1.0, 100 + 1)
y = np.array(m1.historian_df["q1.Rectifier.t1.outputs"])
# y2 = np.array(m1.historian_df["q1.VanDerPol.t1.x1"])
# y2 = np.array(m1.historian_df["q1.BouncingBall2.t1.h"])
# y3 = np.array(m1.historian_df["q1.BouncingBall3.t1.h"])
t = np.array(m1.historian_df["time"])
ax.plot(t, y)
# ax.plot(t, y2)
# ax.plot(t, y3)

ax.set(xlabel='time (s)', ylabel='outputs', title='BB')
ax.grid()

plt.show()

print("execution finished")
