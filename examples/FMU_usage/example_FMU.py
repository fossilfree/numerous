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
        fmu_filename = '/home/artem/fmus/Rectifier.fmu'
        fmu_subsystem = FMU_Subsystem(fmu_filename, "Rectifier", debug_output=True)
        self.register_items([fmu_subsystem])


subsystem1 = S3('q1')
m1 = Model(subsystem1, use_llvm=True)
s = Simulation(
    m1, t_start=0, t_stop=0.1, num=1000, num_inner=1, max_step=.1, solver_type=SolverType.SOLVER_IVP)

s.solve(run_fmu_event_action=True)


fig, ax = plt.subplots()
# y = np.array(m1.historian_df["q1.Rectifier.t1.Rectifier1_Capacitor1_v"])
y1 = np.array(m1.historian_df["q1.Rectifier.t1.outputs"])
t = np.array(m1.historian_df["time"])
# ax.plot(t, y)
ax.plot(t, y1)


ax.set(xlabel='time (s)', ylabel='outputs', title='Rectifier')
ax.grid()

plt.show()

print("execution finished")
