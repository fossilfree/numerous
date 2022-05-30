from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
import numpy as np


class SuperSimpleStepTest(Item, EquationBase):
    def __init__(self, tag='simplestepping'):
        super(SuperSimpleStepTest, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x_dot = 0.1


class System(Subsystem):
    def __init__(self, tag='sys'):
        super().__init__(tag)
        i = SuperSimpleStepTest()
        self.register_item(i)


sys = System()
model = Model(sys)
t_stop = 100
sim = Simulation(model, t_start=0, t_stop=t_stop, num=50, max_step=np.inf, method='RK45')

finished = False
dt = 2
while not finished:
    t, info = sim.step_solve(dt,1)
    print(t)
    if t >= t_stop:
        finished = True

sim.model.create_historian_df()

df = sim.model.historian_df
print(df['time'].values, df['sys.simplestepping.t1.x'].values)
