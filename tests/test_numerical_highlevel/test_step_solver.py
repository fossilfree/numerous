import pytest
from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem, ConnectorTwoWay, ItemsStructure

import numpy as np

if __name__ == "__main__":
    from numerous.engine import model, simulation
    from time import time
    from matplotlib import pyplot as plt


class ThermalMass(EquationBase, Item):
    """
        Equation and item modelling a spring and dampener
    """

    def __init__(self, tag="tm", c=100, T0=20):
        super(ThermalMass, self).__init__(tag)

        # define variables

        self.add_constant('c', c)
        self.add_parameter('P', 0)
        self.add_state('T', T0)

        # define namespace and add equation
        thermal = self.create_namespace('thermal')
        thermal.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T_dot = scope.P/scope.c + scope.T*0


class ThermalConductivityEq(EquationBase):
    def __init__(self, h=1):
        super().__init__(tag='tc')

        self.add_parameter('h', h)  #

        self.add_parameter('T1', 0)  # [kg/s]      Mass flow rate in one side of the valve
        self.add_parameter('T2', 0)  # [kg/s]      Mass flow rate in the other side of the valve
        self.add_parameter('P1', 0)  # [m]         Liquid height in the tank 1 connected to the valve (top tank)
        self.add_parameter('P2', 0)  # [m]         Liquid height in the tank 2 connected to the valve (bottom tank)

        self.add_parameter('P', 0)
        self.add_parameter('dT', 0)

    @Equation()
    def eval(self, scope):

        scope.dT = scope.T1 - scope.T2
        scope.P = scope.h * scope.dT

        scope.P1 = -scope.P
        scope.P2 = scope.P



class ThermalConductivity(ConnectorTwoWay):
    def __init__(self, tag="tmc", h=1):
        super().__init__(tag, side1_name='side1', side2_name='side2')

        # 1 Create a namespace for mass flow rate equation and add the valve equation
        thermal = self.create_namespace('thermal')
        thermal.add_equations([ThermalConductivityEq(h=h)])

        # 2 Create variables H and mdot in side 1 adn 2
        # (side 1 represents a connection with one tank, with related liquid height H_1)
        # (side 1 represents a connection with the second tank, with related liquid height H_2)
        self.side1.thermal.create_variable(name='P')
        self.side1.thermal.create_variable(name='T')

        self.side2.thermal.create_variable(name='P')
        self.side2.thermal.create_variable(name='T')

        # Map variables between binding and internal variables for side 1 and 2
        # This is needed to update the values of the variables in the binding according to the equtions of the items
        thermal.T1 = self.side1.thermal.T
        thermal.T2 = self.side2.thermal.T

        #TODO: Test is failing because these mappings are missed by the engine
        self.side1.thermal.P += thermal.P1
        self.side2.thermal.P += thermal.P2

class ThermalSystem(Subsystem):
    def __init__(self, tag, T0=[50, 150], n=1):
        super().__init__(tag)
        thermalmasses = []
        thermalconductors = []
        last_tm = None
        for i in range(n):
            # Create oscillator
            thermalmass = ThermalMass('thermalmass' + str(i), T0=T0[i])
            thermalmasses.append(thermalmass)

        self.register_items(thermalmasses, tag="thermalmasses", structure=ItemsStructure.SET)

        for i in range(n):
            if i>0:
                tc = ThermalConductivity('tc'+str(i))
                tc.bind(side1=thermalmasses[i-1], side2=thermalmasses[i])
                #tc.side1.thermal.P += tc.thermal.P1
                #tc.side2.thermal.P += tc.thermal.P2
                thermalconductors.append(tc)

        # Register the items to the subsystem to make it recognize them.
        self.register_items(thermalconductors, tag="conductors", structure=ItemsStructure.SET)


#def test_mapping():
if __name__ == '__main__':

    from numerous.engine import model, simulation
    from time import time
    import numpy as np
    n=2
    T0=[150, 50]
    subsystem = ThermalSystem('system', n=n, T0=T0)
    # Define simulation
    n=100
    s = simulation.Simulation(
        model.Model(subsystem),
        t_start=0, t_stop=500.0, num=n, num_inner=1, max_step=1000
    )
    # Solve and plot

    time_ = np.logspace(0,4,n)
    s.reset()

    t_last = 0
    tic = time()
    for t_ in time_:
        #print('time: ',t_)
        stop = s.step(t_-t_last)
        if stop:
            break
        t_last=t_
    toc = time()
    s.complete()
    print('solve time: ', toc-tic)
    print(s.model.historian_df['time'])





