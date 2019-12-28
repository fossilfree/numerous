import random
import time
import sys

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Item, ConnectorTwoWay, Subsystem
from numerous import Equation
from numerous.engine import OverloadAction
from numerous.multiphysics import equation


class Thermal_Conductance_Equation(Equation):
    """
    Equation for a simple thermal conductor
    """
    def __init__(self, k=100):
        super().__init__(tag='thermal_capacity_equation')

        self.add_constant('k', k)
        self.add_parameter('T1', 0)
        self.add_parameter('T2', 0)
        self.add_parameter('P1', 0)
        self.add_parameter('P2', 0)

    @equation
    def eval(self, scope):
        P = (scope.T1 - scope.T2) * scope.k
        scope.P1 = -P
        scope.P2 = P


class Thermal_Capacitance(Equation, Item):
    """
        Equation and item modelling a thermal capacitance
    """
    def __init__(self, tag="tm", C=1000, T0=0):
        super(Thermal_Capacitance, self).__init__(tag)

        self.add_constant('C', C)
        self.add_parameter('P', 0)
        self.add_state('T', T0)

        thermal_transport = self.create_namespace('thermal_transport')
        thermal_transport.add_equations([self],
                                        on_assign_overload=OverloadAction.SUM)

    @equation
    def eval(self, scope):
        scope.T_dot = scope.P / scope.C


class Thermal_Conductor(ConnectorTwoWay):
    def __init__(self, tag="tm", k=100):
        super().__init__(tag, side1_name='side1', side2_name='side2')

        #Create a namespace for thermal transport equations
        thermal_transport = self.create_namespace('thermal_transport')
        #Add the the thermal conductance equation
        thermal_transport.add_equations([Thermal_Conductance_Equation(k=k)])

        #Create variables T and P in side 1 binding - this is so we now the item we later bind will have these variable
        self.side1.thermal_transport.create_variable(name='T')
        self.side1.thermal_transport.create_variable(name='P')

        #Map variables between binding and internal variables for side 1 - this is so the variables of the binding side 1 item will be updated based on operations in the equation of the item
        thermal_transport.T1 = self.side1.thermal_transport.T
        self.side1.thermal_transport.P = thermal_transport.P1

        # Create variables T and P in side 2 binding - this is so we now the item we later bind will have these variable
        self.side2.thermal_transport.create_variable(name='T')
        self.side2.thermal_transport.create_variable(name='P')

        # Map variables between binding and internal variables for side 2 - this is so the variables of the binding side 2 item will be updated based on operations in the equation of the item
        thermal_transport.T2 = self.side2.thermal_transport.T
        self.side2.thermal_transport.P = thermal_transport.P2


class ThermalCapacitancesSeries(Subsystem):
    def __init__(self, tag, T0):
        super().__init__(tag)

        items = []

        prev_node = None
        #Create N heat capacitances and connect them.
        for i, T0_ in enumerate(T0):


            #Create thermal conductor
            node = Thermal_Capacitance('node' + str(i), C=100, T0=T0_)

            items.append(node)
            if prev_node:
                # Connect the last node to the new node with a conductor
                thermal_conductor = Thermal_Conductor('thermal_conductor' + str(i), k=1)
                thermal_conductor.bind(side1=prev_node, side2=node)
                #Append the thermal conductor to the item.
                items.append(thermal_conductor)

            prev_node = node

        #Register the items to the subsystem to make it recognize them.
        self.register_items(items)


if __name__ == "__main__":
    # Create a model with three nodes
    X= []
    Y= []
    Z= []
    for i in range(1, int(sys.argv[1]), int(sys.argv[2])):
        T0 = [random.randrange(1, 101, 1) for _ in range(i)]
        m = Model(ThermalCapacitancesSeries("tcs", T0))
        start = time.time()
        # Define simulation
        s = Simulation(m, t_start=0, t_stop=1, num=10, num_inner=100, max_step=0.1)
        #
        # solve simulation
        s.solve()
        end = time.time()
        # print(m.states_as_vector)
        # print some statitics and info
        print(m.info)
        X.append(i)
        Z.append(m.info['Assemble time'])
        Y.append(end-start)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(X, Y, label='solve')
    ax.plot(X, Z, label='assemble')
    plt.legend(loc="upper left")
    plt.xlabel("number of objects")
    plt.ylabel("seconds")
    plt.show()

