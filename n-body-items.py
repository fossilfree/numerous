from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system.item import Item
import numpy as np
import matplotlib.pyplot as plt


from numerous.engine.system import EquationBase, Subsystem
from numerous.multiphysics import Equation


def plot(bodies):
    fig = plt.figure(figsize=(10, 10))
    subplot = fig.add_subplot(projection='3d')
    colors = ['aqua', 'aquamarine', 'black', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'grey', 'hotpink', 'indianred', 'indigo', 'lawngreen', 'lightcoral', 'lightgreen', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lime', 'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegreen', 'palevioletred', 'peru', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'springgreen', 'steelblue', 'tan', 'teal', 'tomato', 'turquoise', 'yellow', 'yellowgreen']
    colorcount=len(colors)
    for n,i in enumerate(bodies):
        color=colors[n%colorcount]
        print(n, color)
        x, y, z = i
        sublists = []
        for i in range(len(x)):
            current = np.array([x[i], y[i], z[i]])
            sublists.append(current)
        r = np.array(sublists)

        subplot.plot(r[:, 0], r[:, 1], r[:, 2], color)
        subplot.plot([r[0, 0]], [r[0, 1]], [r[0, 2]], 'ko')

    plt.legend(['Trajectory', 'Starting Position'])
    plt.show()


earth_radius = 6378.0
earth_mu = 398600.0

# universal gravitation constant
# G = 6.67408e-11 (m**3/kg/s**2)
G = 6.67259e-20  # (km**3/kg/s**2)


class Oribtal(EquationBase, Item):
    def __init__(self, initial, bodies, G, tag='orbit'):
        super(Oribtal, self).__init__(tag)
        self.add_constant('bodies',bodies)
        self.add_constant('G', G)
        for i in range(bodies):
            self.add_state(f'mu_{i}', initial[0+i*7])
            self.add_state(f'rx_{i}', initial[1+i*7])
            self.add_state(f'ry_{i}', initial[2+i*7])
            self.add_state(f'rz_{i}', initial[3+i*7])
            self.add_state(f'vx_{i}', initial[4+i*7])
            self.add_state(f'vy_{i}', initial[5+i*7])
            self.add_state(f'vz_{i}', initial[6+i*7])
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def diffy_q(self, scope):
        G = scope.G

        print(scope)

        for b1 in range(scope.bodies - 1, -1, -1):
            for b2 in range(scope.bodies - 1, -1, -1):
                if b1 != b2:
                    rx = scope['rx_'+str(b2)] - scope['rx_'+str(b1)]
                    ry = scope['ry_'+str(b1)] - scope['ry_'+str(b1)]
                    rz = scope['rz_'+str(b2)] - scope['rz_'+str(b1)]

                    # normalize differences
                    norm_r = (rx ** 2 + ry ** 2 + rz ** 2) ** (1 / 2)

                    #scope[f'vx_{b1}_dot'] = G * scope[f'mu_{b2}'] * (scope[f'rx_{b2}'] - scope[f'rx_{b1}']) / norm_r ** 3 + scope[f'vx_{b1}_dot']
                    #scope[f'vy_{b1}_dot'] = G * scope[f'mu_{b2}'] * (scope[f'ry_{b2}'] - scope[f'ry_{b1}']) / norm_r ** 3 + scope[f'vy_{b1}_dot']
                    #scope[f'vz_{b1}_dot'] = G * scope[f'mu_{b2}'] * (scope[f'rz_{b2}'] - scope[f'rz_{b1}']) / norm_r ** 3 + scope[f'vz_{b1}_dot']

            #scope[f'rx_{b1}_dot'] = scope[f'vx_{b1}']
            #scope[f'ry_{b1}_dot'] = scope[f'vy_{b1}']
            #scope[f'rz_{b1}_dot'] = scope[f'vz_{b1}']

class Nbody(Subsystem):
    def __init__(self, initial, bodies, G, tag="nbody"):
        super().__init__(tag)
        self.register_item(Oribtal(initial=initial, bodies=bodies, G=G))


if __name__ == '__main__':
    r_mag = earth_radius + 500.0
    v_mag = np.sqrt(earth_mu / r_mag)

    inital_bodies = [
        [
            [-10, 0, 0], [0, 1, 1], 1e20
        ], [
            [0, 0, 0], [0, 1, 0], 1e20
        ], [
            [10, 0, 0], [0, 1, -1], 1e20
        ]
    ]

    G=6.67259e-20

    tspan = 100 * 60.0

    dt = 100.0

    n_steps = int(np.ceil(tspan / dt))
    y0 = []
    for i in inital_bodies:
        y0 += [i[2]] + i[0] + i[1]
    print(y0)
    print(f'Running with {len(inital_bodies)} bodies')
    nbody_system = Nbody(initial=y0, bodies=len(inital_bodies), G=G)
    nbody_model = Model(nbody_system, use_llvm=False)
    nbody_simulation = Simulation(nbody_model, t_start=0, t_stop=1000000, num=1000,
                                 max_step=1, method="Euler")
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

    bodies = [[x_0, y_0, z_0], [x_1, y_1, z_1], [x_2, y_2, z_2]]

    plot(bodies)
