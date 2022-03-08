from numerous import Equation
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation, SolverType
from numerous.engine.system.item import Item
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D

from numerous.engine.system import EquationBase, Subsystem


def plot(r):
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

class Oribtal(EquationBase, Item):
    def __init__(self,  initial, mu, tag='orbit'):
        super(Oribtal, self).__init__(tag)
        self.add_parameter('mu',mu)
        self.add_state('rx',initial[0])
        self.add_state('ry',initial[1])
        self.add_state('rz',initial[2])
        self.add_state('vx',initial[3])
        self.add_state('vy',initial[4])
        self.add_state('vz',initial[5])
        self.add_state('ax')
        self.add_state('ay')
        self.add_state('az')
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])
    @Equation()
    def diffy_q(self,scope):
        r=np.array([scope.rx,scope.ry,scope.rz])
        norm_r = (r[0]**2+r[1]**2+r[2]**2)**(1/2)
        # norm_r=np.linalg.norm(r)

        scope.ax,scope.ay,scope.az=-r*scope.mu/norm_r**3

class Nbody(Subsystem):
    def __init__(self, initial, mu, tag="nbody"):
        super().__init__(tag)
        self.register_item(Oribtal(initial=initial, mu=mu))

if __name__ == '__main__':
    r_mag=earth_radius+500.0
    v_mag=np.sqrt(earth_mu/r_mag)

    r0=[r_mag,0,0]
    v0=[0,v_mag,0]

    tspan=100*60.0

    dt=100.0

    n_steps=int(np.ceil(tspan/dt))
    y0=r0+v0
    nbody_system = Nbody(initial=y0, mu=earth_mu)
    nbody_model = Model(nbody_system,use_llvm=False)
    nbody_simulation = Simulation(nbody_model, solver_type=SolverType.SOLVER_IVP, t_start=0, t_stop=10.0, num=100, num_inner=100, max_step=1)
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
    #     step+=1
    #
    # rs=ys[:,:3]
    rs = nbody_simulation.model.historian_df[['nbody.orbital.mechanics.rx',
                                              'nbody.orbital.mechanics.ry',
                                              'nbody.orbital.mechanics.rz']]

    plot(rs)