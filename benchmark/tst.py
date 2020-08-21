import random
import time


from numerous.engine.model import  Model
from numerous.engine.simulation import Simulation
from numerous.engine.simulation.solvers.base_solver import SolverType
from numerous.engine.system import Item, ConnectorTwoWay, Subsystem

from numerous import EquationBase
from numerous.multiphysics import Equation
import os

def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)


def kill_numba_cache():

    root_folder = os.path.realpath(__file__ + "/../")

    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                except Exception as e:
                    print("failed on %s", root)

class Thermal_Conductance_Equation(EquationBase):
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

    @Equation()
    def eval(self,scope):
        P = (scope.T1 - scope.T2) * scope.k
        scope.P1 = -P
        scope.P2 = P


class Thermal_Capacitance(EquationBase, Item):
    """
        Equation and item modelling a thermal capacitance
    """
    def __init__(self, tag="tm", C=1000, T0=0):
        super(Thermal_Capacitance, self).__init__(tag)

        self.wrapper_ = 1
        self.add_constant('C', C)
        self.add_parameter('P', 0)
        self.add_state('T', T0)

        thermal_transport = self.create_namespace('thermal_transport')
        thermal_transport.add_equations([self])

    @Equation()
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
    def __init__(self, tag, Tinit=100, T0=25, num_nodes=10, k=1):
        super().__init__(tag)

        items = []


        #Create N heat capacitances and connect them.

        inlet_node = Thermal_Capacitance('node0', C=100, T0=Tinit)
        items.append(inlet_node)
        prev_node = inlet_node
        for i in range(1,num_nodes):
            import pickle as pickle


            #Create thermal conductor
            node = Thermal_Capacitance('node' + str(i), C=100, T0=T0)
            # Connect the last node to the new node with a conductor
            thermal_conductor = Thermal_Conductor('thermal_conductor' + str(i), k=k)
            thermal_conductor.bind(side1=prev_node, side2=node)
            #Append the thermal conductor to the item.
            items.append(thermal_conductor)
            items.append(node)
            prev_node = node

        #Register the items to the subsystem to make it recognize them.
        self.register_items(items)

def timeit(s):
    # kill_numba_cache()
    start = time.time()
    s.solve()
    end = time.time()
    dt = end - start
    print(dt)
    return dt

if __name__ == "__main__":
    import resource
    import sys

    # print(resource.getrlimit(resource.RLIMIT_STACK))
    # print(sys.getrecursionlimit())

    max_rec = 0x100000

    # May segfault without this line. 0x100 is a guess at the size of each stack frame.
    resource.setrlimit(resource.RLIMIT_STACK, [0x1000 * max_rec, resource.RLIM_INFINITY])
    sys.setrecursionlimit(max_rec)
    # Create a model with three nodes

    X = []
    Y = []
    Z = []

    num_nodes =[100,1000,10000]
    Tinit = 100
    T0 = 25
    k = 1

    solver_type = SolverType.NUMEROUS
    for i in num_nodes:#range(1,num_nodes+1):

        m = Model(ThermalCapacitancesSeries("tcs", num_nodes=i, Tinit=Tinit, T0=T0, k=k))


        # print(m.states_as_vector)
        # Define simulation
        s = Simulation(m, t_start=0, t_stop=100, num=100, num_inner=1, max_step=0.1, solver_type=solver_type)
        #
        # solve simulation
        dt = timeit(s)

    # print(m.states_as_vector)
    # print some statitics and info
    # print(m.states_as_vector)
    #print(m.info)

        X.append(i)
        Z.append(m.info['Assemble time'])
        Y.append(dt)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(X, Y, label='solve')
    #ax.plot(X, Z, label='assemble')
    plt.legend(loc="upper left")
    plt.xlabel("number of objects")
    plt.ylabel("seconds")
    plt.title(f'SolverType: {solver_type}')
    plt.show()
