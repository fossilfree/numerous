from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem, ConnectorTwoWay, ItemsStructure

import numpy as np

if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    from matplotlib import pyplot as plt

class DampenedOscillator(EquationBase, Item):
    """
        Equation and item modelling a spring and dampener
    """
    def __init__(self, tag="tm", x0=1, k=1, c=1, a=1):
        super(DampenedOscillator, self).__init__(tag)

        #define variables
        self.add_constant('k', k)
        self.add_constant('c', c)
        self.add_constant('a', a)
        self.add_state('x', x0)
        self.add_state('v', 0)
        #self.add_state('v2', 0)

        #define namespace and add equation
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        #scope.c = -1 + 1
        #z = 4 + 2
        scope.a  =  -scope.k * scope.x - scope.c * scope.v
        #scope.a = scope.c * scope.v
        scope.v_dot = scope.a
        scope.x_dot = scope.v
        #a1 = 2 * scope.x_dot
        #scope.v2_dot = scope.x_dot

class Spring_Equation(EquationBase):
    def __init__(self, k=1, dx0=1):
        super().__init__(tag='spring_equation')

        self.add_parameter('k', k)  #
        self.add_parameter('c', 0)  #
        self.add_parameter('dx0', dx0)  #
        self.add_parameter('F1', 0)  # [kg/s]      Mass flow rate in one side of the valve
        self.add_parameter('F2', 0)  # [kg/s]      Mass flow rate in the other side of the valve
        self.add_parameter('x1', 0)  # [m]         Liquid height in the tank 1 connected to the valve (top tank)
        self.add_parameter('x2', 0)  # [m]         Liquid height in the tank 2 connected to the valve (bottom tank)



    @Equation()
    def eval(self, scope):
        scope.c = scope.k
        dx = scope.x1 - scope.x2
        F = np.abs(dx)*scope.c

        scope.F1 = -F if scope.x1 > scope.x2 else F # [kg/s]

        scope.F2 = -scope.F1

class TestEq(EquationBase, Item):
    def __init__(self, k=2):
        super().__init__(tag='spring_equation')

        self.add_parameter('k', k)  #
        # define namespace and add equation
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])
    # Define the valve as a connector item - connecting two tanks
class SpringCoupling(ConnectorTwoWay):
    def __init__(self, tag="springcoup", k=1, dx0=0):
        super().__init__(tag, side1_name='side1', side2_name='side2')

        # 1 Create a namespace for mass flow rate equation and add the valve equation
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([Spring_Equation(k=k, dx0=dx0)])


        # 2 Create variables H and mdot in side 1 adn 2
        # (side 1 represents a connection with one tank, with related liquid height H_1)
        # (side 1 represents a connection with the second tank, with related liquid height H_2)
        self.side1.mechanics.create_variable(name='v_dot')
        self.side1.mechanics.create_variable(name='x')

        self.side2.mechanics.create_variable(name='v_dot')
        self.side2.mechanics.create_variable(name='x')


        # Map variables between binding and internal variables for side 1 and 2
        # This is needed to update the values of the variables in the binding according to the equtions of the items
        mechanics.x1 = self.side1.mechanics.x
        mechanics.x2 = self.side2.mechanics.x

        #self.side1.mechanics.v_dot += mechanics.F1
        #self.side2.mechanics.v_dot += mechanics.F2

class OscillatorSystem(Subsystem):
    def __init__(self, tag, c=1, k=1, x0=[10, 8], a=1, n=1):
        super().__init__(tag)
        oscillators = []
        for i in range(n):
            #Create oscillator
            oscillator = DampenedOscillator('oscillator'+str(i), k=k, c=c, x0=x0[i], a=a)
            oscillators.append(oscillator)

        self.register_items(oscillators, tag="oscillators", structure=ItemsStructure.SET)
        # 3. Valve_1 is one instance of valve class

        if True:#len(oscillators)>1:
            spc1 = SpringCoupling('spc1', k=.0001, dx0=4)
            spc1.bind(side1=oscillators[0], side2=oscillators[1])
            spc1.side1.mechanics.v_dot += spc1.mechanics.F1
            spc1.side2.mechanics.v_dot += spc1.mechanics.F2

            spc2 = SpringCoupling('spc2', k=.001, dx0=4)
            spc2.bind(side1=oscillators[0], side2=oscillators[1])
            spc2.side1.mechanics.v_dot += spc2.mechanics.F1
            spc2.side2.mechanics.v_dot += spc2.mechanics.F2

            #Register the items to the subsystem to make it recognize them.
            self.register_items([spc1, spc2], tag="couplings", structure=ItemsStructure.SET)

            spc3 = SpringCoupling('spc3', k=.001, dx0=4)
            spc3.bind(side1=oscillators[0], side2=oscillators[1])
            spc3.side1.mechanics.v_dot += spc3.mechanics.F1
            spc3.side2.mechanics.v_dot += spc3.mechanics.F2



            # Register the items to the subsystem to make it recognize them.
            self.register_items([spc3])
            a=1


if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    from matplotlib import pyplot as plt

    # Define simulation
    s = simulation.Simulation(
        model.Model(OscillatorSystem('system', k=0.01,  c=0.001, a=0, n=2, x0=[1,2,3])),
        t_start=0, t_stop=500.0, num=1000, num_inner=100, max_step=1
    )
    # Solve and plot
    tic = time()
    s.solve()
    toc = time()
    print('Execution time: ', toc-tic)
    #print(s.model.historian_df)
    #print(len(list(s.model.historian_df)))
    #s.model.historian_df['oscillator0_mechanics_a'].plot()
    #for i in range(10):
    #    for k, v in zip(list(s.model.historian_df),s.model.historian_df.loc[i,:]):
    #        print(k,': ',v)


    #print(s.model.historian_df.describe())
    print(list(s.model.historian_df))
    #for c in list(s.model.historian_df):
    #    if not c == 'time':
            #print(s.model.historian_df[c].describe())
    print(list(s.model.historian_df))
    s.model.historian_df[['system.oscillator0.mechanics.x', 'system.oscillator1.mechanics.x']].plot()
    #print()
    plt.show()
    plt.interactive(False)