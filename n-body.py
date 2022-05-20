from numerous import Equation
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation, SolverType
from numerous.engine.system.item import Item
import numpy as np
import matplotlib.pyplot as plt

from numerous.engine.system import EquationBase, Subsystem


def plot(bodies):
    fig = plt.figure(figsize=(10, 10))
    subplot = fig.add_subplot(projection='3d')
    for i in bodies:
        x,y,z=i
        sublists=[]
        for i in range(len(x)):
            current=np.array([x[i],y[i],z[i]])
            sublists.append(current)
        r=np.array(sublists)

        subplot.plot(r[:,0],r[:,1],r[:,2],'k')
        subplot.plot([r[0,0]],[r[0,1]],[r[0,2]],'ko')


    plt.legend(['Trajectory','Starting Position'])
    plt.show()

earth_radius=6378.0
earth_mu=398600.0

# universal gravitation constant
# G = 6.67408e-11 (m**3/kg/s**2)
G = 6.67259e-20  # (km**3/kg/s**2)

class Oribtal(EquationBase, Item):
    def __init__(self,  initial, mu, tag='orbit'):
        super(Oribtal, self).__init__(tag)
        self.add_parameter('mu',mu)
        self.add_state('rx_0',initial[0])
        self.add_state('ry_0',initial[1])
        self.add_state('rz_0',initial[2])
        self.add_state('vx_0',initial[3])
        self.add_state('vy_0',initial[4])
        self.add_state('vz_0',initial[5])
        self.add_state('rx_1', initial[6])
        self.add_state('ry_1', initial[7])
        self.add_state('rz_1', initial[8])
        self.add_state('vx_1', initial[9])
        self.add_state('vy_1', initial[10])
        self.add_state('vz_1', initial[11])
        self.add_state('rx_2', initial[12])
        self.add_state('ry_2', initial[13])
        self.add_state('rz_2', initial[14])
        self.add_state('vx_2', initial[15])
        self.add_state('vy_2', initial[16])
        self.add_state('vz_2', initial[17])
        self.add_state('step', 0)
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def diffy_q(self, scope):
        scope.step = -1

        G = 6.67259e-20
        m_0 = 1e26
        m_1 = 1e26
        m_2 = 1e26

        scope.step = 1

        #create pos diff
        rx10 = scope.rx_1 - scope.rx_0
        ry10 = scope.ry_1 - scope.ry_0
        rz10 = scope.rz_1 - scope.rz_0

        scope.step = 2

        rx21 = scope.rx_2 - scope.rx_1
        ry21 = scope.ry_2 - scope.ry_1
        rz21 = scope.rz_2 - scope.rz_1

        scope.step = 3

        rx20 = scope.rx_2 - scope.rx_0
        ry20 = scope.ry_2 - scope.ry_0
        rz20 = scope.rz_2 - scope.rz_0

        scope.step = 4

        #normalize differences
        norm_r10 = (rx10 ** 2 + ry10 ** 2 + rz10 ** 2) ** (1 / 2)
        norm_r21 = (rx21 ** 2 + ry21 ** 2 + rz21 ** 2) ** (1 / 2)
        norm_r20 = (rx20 ** 2 + ry20 ** 2 + rz20 ** 2) ** (1 / 2)

        scope.step = 5

        vx_0_dot = 0
        vy_0_dot = 0
        vz_0_dot = 0

        vx_1_dot = 0
        vy_1_dot = 0
        vz_1_dot = 0

        vx_2_dot = 0
        vy_2_dot = 0
        vz_2_dot = 0

        #compute 10
        vx_0_dot += G * m_1 * (scope.rx_1 - scope.rx_0) / norm_r10 ** 3

        vx_1_dot += G * m_0 * (scope.rx_0 - scope.rx_1) / norm_r10 ** 3

        vy_0_dot += G * m_1 * (scope.ry_1 - scope.ry_0) / norm_r10 ** 3

        vy_1_dot += G * m_0 * (scope.ry_0 - scope.ry_1) / norm_r10 ** 3

        vz_0_dot += G * m_1 * (scope.rz_1 - scope.rz_0) / norm_r10 ** 3

        vz_1_dot += G * m_0 * (scope.rz_0 - scope.rz_1) / norm_r10 ** 3

        scope.step = 6

        #compute 21
        vx_1_dot += G * m_2 * (scope.rx_2 - scope.rx_1) / norm_r21 ** 3

        vx_2_dot += G * m_1 * (scope.rx_1 - scope.rx_2) / norm_r21 ** 3

        vy_1_dot += G * m_2 * (scope.ry_2 - scope.ry_1) / norm_r21 ** 3

        vy_2_dot += G * m_1 * (scope.ry_1 - scope.ry_2) / norm_r21 ** 3

        vz_1_dot += G * m_2 * (scope.rz_2 - scope.rz_1) / norm_r21 ** 3

        vz_2_dot += G * m_1 * (scope.rz_1 - scope.rz_2) / norm_r21 ** 3

        scope.step = 7

        # compute 20
        vx_0_dot += G * m_2 * (scope.rx_2 - scope.rx_0) / norm_r20 ** 3

        vx_2_dot += G * m_0 * (scope.rx_0 - scope.rx_2) / norm_r20 ** 3

        vy_0_dot += G * m_2 * (scope.ry_2 - scope.ry_0) / norm_r20 ** 3

        vy_2_dot += G * m_0 * (scope.ry_0 - scope.ry_2) / norm_r20 ** 3

        vz_0_dot += G * m_2 * (scope.rz_2 - scope.rz_0) / norm_r20 ** 3

        vz_2_dot += G * m_0 * (scope.rz_0 - scope.rz_2) / norm_r20 ** 3

        scope.step = 8

        scope.vx_0_dot = vx_0_dot
        scope.vy_0_dot = vy_0_dot
        scope.vz_0_dot = vz_0_dot

        scope.vx_1_dot = vx_1_dot
        scope.vy_1_dot = vy_1_dot
        scope.vz_1_dot = vz_1_dot

        scope.vx_2_dot = vx_2_dot
        scope.vy_2_dot = vy_2_dot
        scope.vz_2_dot = vz_2_dot

        scope.rx_0_dot = scope.vx_0
        scope.ry_0_dot = scope.vy_0
        scope.rz_0_dot = scope.vz_0
        scope.rx_1_dot = scope.vx_1
        scope.ry_1_dot = scope.vy_1
        scope.rz_1_dot = scope.vz_1
        scope.rx_2_dot = scope.vx_2
        scope.ry_2_dot = scope.vy_2
        scope.rz_2_dot = scope.vz_2

        scope.step = 9

class Nbody(Subsystem):
    def __init__(self, initial, mu, tag="nbody"):
        super().__init__(tag)
        self.register_item(Oribtal(initial=initial, mu=mu))

if __name__ == '__main__':
    r_mag=earth_radius+500.0
    v_mag=np.sqrt(earth_mu/r_mag)

    inital_bodies=[
        [
            [0.5, 0, 0],[0, 0, 0]
        ],[
            [0, 0.5, 0], [0, 0, 0]
        ],[
            [0, 0, 0.5], [0, 0, 0]
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
    nbody_model = Model(nbody_system,use_llvm=False)
    nbody_simulation = Simulation(nbody_model, solver_type=SolverType.SOLVER_IVP, t_start=0, t_stop=10, num=5 , num_inner=1, max_step=1)
    nbody_simulation.solve()

    x_0 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rx_0"])
    y_0 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.ry_0"])
    z_0 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rz_0"])

    x_1 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rx_1"])
    y_1 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.ry_1"])
    z_1 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rz_1"])

    x_2 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rx_2"])
    y_2 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.ry_2"])
    z_2 = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rz_2"])

    step = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.step"])

    print(len(x_2))
    print(step)
    print(x_0)


    bodies=[[x_0,y_0,z_0],[x_1,y_1,z_1],[x_2,y_2,z_2]]

    # plot(bodies)
