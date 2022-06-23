import numpy as np
from matplotlib import pyplot as plt

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Item, Subsystem
from numerous.engine.system.fmu_subsystem import FMU_Subsystem


class S3(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)

        fmu_filename = '/home/artem/fmus/PCU_RHU_EnergyMachines_01_modex_Linux.fmu'
        fmu_subsystem = FMU_Subsystem(fmu_filename, "PCU_RHU", debug_output=True)
        # fmu_subsystem.t1.variables["h"].value = 19
        self.register_items([fmu_subsystem])


subsystem1 = S3('q1')

m1 = Model(subsystem1, use_llvm=True)
s = Simulation(
    m1, t_start=0, t_stop=0.2, num=100, num_inner=1, max_step=.1)

s.solve()

fig, ax = plt.subplots()

y1 = np.array(m1.historian_df['q1.PCU_RHU.t1.P_out'])
t = np.array(m1.historian_df["time"])
ax.plot(t, y1)

ax.set(xlabel='time (s)', ylabel='outputs', title='Something')
ax.grid()

plt.show()

print("execution finished")
