from numerous import Equation
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation, SolverType
from numerous.engine.system.item import Item
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

from numerous.engine.system import EquationBase, Subsystem


def plot(x,y,z):
    sublists=[]
    for i in range(len(x)):
        current=np.array([x[i],y[i],z[i]])
        sublists.append(current)
    r=np.array(sublists)
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')

    ax.plot(r[:,0],r[:,1],r[:,2],'k')
    ax.plot([r[0,0]],[r[0,1]],[r[0,2]],'ko')

    r_plot=earth_radius

    _u,_v=np.mgrid[0:2*np.pi:20j,0:np.pi:10j]
    _x=r_plot*np.cos(_u)*np.sin(_v)
    _y=r_plot*np.sin(_u)*np.sin(_v)
    _z=r_plot*np.cos(_v)
    ax.plot_surface(_x,_y,_z,cmap='Blues')

    l=r_plot*2.0
    x,y,z=[[0,0,0],[0,0,0],[0,0,0]]
    u,v,w=[[1,0,0],[0,1,0],[0,0,1]]
    ax.quiver(x,y,z,u,v,w,color='k')

    max_val=np.max(np.abs(r))

    ax.set_xlim([-max_val,max_val])
    ax.set_ylim([-max_val,max_val])
    ax.set_zlim([-max_val,max_val])
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_aspect('auto')

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
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def diffy_q(self, scope):
        G = 6.67259e-20
        m_0 = 1e26
        m_1 = 1e26
        rx = scope.rx_1 - scope.rx_0
        ry = scope.ry_1  -scope.ry_0
        rz = scope.rz_1 - scope.rz_0
        norm_r = (rx**2+ry**2+rz**2)**(1/2)


        scope.vx_0_dot =G* m_1 *(scope.rx_1 - scope.rx_0) / norm_r**3

        scope.vx_1_dot = G * m_0 * (scope.rx_0 - scope.rx_1) / norm_r ** 3



        # scope.vy_dot = -scope.ry*scope.mu/norm_r**3
        # scope.vz_dot = -scope.rz*scope.mu/norm_r**3

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

    r0 = [r_mag, 0, 4000]
    v0 = [0, v_mag, 0]

    tspan=100*60.0

    dt=100.0

    n_steps=int(np.ceil(tspan/dt))
    y0=r0+v0
    nbody_system = Nbody(initial=y0, mu=earth_mu)
    nbody_model = Model(nbody_system,use_llvm=False)
    nbody_simulation = Simulation(nbody_model, solver_type=SolverType.SOLVER_IVP, t_start=0, t_stop=20000.0, num=100, num_inner=100, max_step=1)
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
    rs = nbody_simulation.model.historian_df['nbody.orbit.mechanics.rx']
    x = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rx"])
    y = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.ry"])
    z = np.array(nbody_simulation.model.historian_df["nbody.orbit.mechanics.rz"])

    plot(x,y,z)
