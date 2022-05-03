from numerous import Equation
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation, SolverType
from numerous.engine.system.item import Item
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

from numerous.engine.system import EquationBase, Subsystem


def plot(bodies):
    plots=[]
    pl_index=0
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(projection='3d')
    x,y,z=[[0,0,0],[0,0,0],[0,0,0]]
    u,v,w=[[1,0,0],[0,1,0],[0,0,1]]
    for i in bodies:
        x,y,z=i
        sublists=[]
        for i in range(len(x)):
            current=np.array([x[i],y[i],z[i]])
            sublists.append(current)
        r=np.array(sublists)

        #plots.append()

        plt.plot(r[:,0],r[:,1],r[:,2],'k')
        plt.plot([r[0,0]],[r[0,1]],[r[0,2]],'ko')

        #plots[pl_index].plot(r[:,0],r[:,1],r[:,2],'k')
        #plots[pl_index].plot([r[0,0]],[r[0,1]],[r[0,2]],'ko')

        pl_index += 1

    plt.legend(['Trajectory','Starting Position'])
    plt.show()

earth_radius=6378.0
earth_mu=398600.0

# universal gravitation constant
# G = 6.67408e-11 (m**3/kg/s**2)
G = 6.67259e-20  # (km**3/kg/s**2)

class Oribtal(EquationBase, Item):
    def __init__(self,  initial, mu, tag='orbit', bodies=2):
        super(Oribtal, self).__init__(tag)
        self.add_parameter('mu',mu)
        self.add_parameter('bodies', bodies)
        init_offset=0
        for i in range(bodies):
            self.add_state(f'rx_{i}', initial[init_offset])
            self.add_state(f'ry_{i}', initial[init_offset+1])
            self.add_state(f'rz_{i}', initial[init_offset+2])
            self.add_state(f'vx_{i}', initial[init_offset+3])
            self.add_state(f'vy_{i}', initial[init_offset+4])
            self.add_state(f'vz_{i}', initial[init_offset+5])
            self.add_state(f'm_{i}', initial[init_offset+6])
            init_offset += 7
        self.add_state('stage', 0)
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def diffy_q(self, scope):
        G = 6.67259e-20
        m_0 = 1e26
        m_1 = 1e26
        rx = scope.rx_1 - scope.rx_0
        ry = scope.ry_1 - scope.ry_0
        rz = scope.rz_1 - scope.rz_0
        norm_r = (rx ** 2 + ry ** 2 + rz ** 2) ** (1 / 2)

        scope.vx_0_dot = G * m_1 * (scope.rx_1 - scope.rx_0) / norm_r ** 3

        scope.vx_1_dot = G * m_0 * (scope.rx_0 - scope.rx_1) / norm_r ** 3

        scope.rx_0_dot = scope.vx_0
        scope.ry_0_dot = scope.vy_0
        scope.rz_0_dot = scope.vz_0
        scope.rx_1_dot = scope.vx_1
        scope.ry_1_dot = scope.vy_1
        scope.rz_1_dot = scope.vz_1

class Nbody(Subsystem):
    def __init__(self, initial, mu, tag="nbody"):
        super().__init__(tag)
        self.register_item(Oribtal(initial=initial, mu=mu))

if __name__ == '__main__':
    r_mag=earth_radius+500.0
    v_mag=np.sqrt(earth_mu/r_mag)

    inital_bodies=[
        [
            [10, 0, 0],[0, 0, 0]
        ],[
            [0, 10, 0], [0, 0, 0]
        ],[
            [0, 0, 10], [0, 0, 0]
        ]
    ]

    tspan=100*60.0

    dt=100.0

    n_steps=int(np.ceil(tspan/dt))
    y0=[]
    for i in inital_bodies:
        y0+=i[0]+i[1]
    print(y0)
    nbody_system = Nbody(initial=y0, mu=earth_mu)
    nbody_model = Model(nbody_system,use_llvm=True)
    nbody_simulation = Simulation(nbody_model, solver_type=SolverType.SOLVER_IVP, t_start=0, t_stop=2000000.0, num=10000, num_inner=100, max_step=1)
    nbody_simulation.solve()
    # ys=np.zeros((n_steps,6))
    # ts=np.zeros((n_steps,1))
    #
    #
    #
    # ys[0]=y0
    # step=1
    #
    # while solver.successful() and step<n_steps:
    #     solver.integrate(solver.t+dt)
    #     ts[step]=solver.t
    #     ys[step]=solver.y
    #     step+=1plot
    #
    # rs=ys[:,:3]
    #rs = nbody_simulation.model.historian_df['nbody.orbit.mechanics.rx']
    x_0 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rx_0"])
    y_0 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.ry_0"])
    z_0 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rz_0"])

    x_1 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rx_1"])
    y_1 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.ry_1"])
    z_1 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rz_1"])

    stage = list(nbody_simulation.model.historian_df["nbody.orbit.mechanics.stage"])

    bodies=[[x_0,y_0,z_0],[x_1,y_1,z_1]]
    for p, i in enumerate(bodies):
        for j in i:
            print(len(j))
    print(stage)

    plot(bodies)
