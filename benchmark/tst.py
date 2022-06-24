import plotly.graph_objects as go
import time


from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.simulation.solvers.base_solver import SolverType
from numerous.engine.system import Item, ConnectorTwoWay, Subsystem

from numerous.multiphysics import EquationBase, Equation

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
    def eval(self, scope):
        P = (scope.T1 - scope.T2) * scope.k
        #print(scope.T1, scope.T2)
        # print(global_variables.time)
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

        thermal_capacitance = self.create_namespace('thermal_capacitance')
        thermal_capacitance.add_equations([self])

    @Equation()
    def eval(self, scope):
#        print(scope.C, scope.P, scope.T)

        scope.T_dot = scope.P / scope.C


class Thermal_Conductor(Subsystem):
    def __init__(self, tag="tm", k=100, side1=None, side2=None):
        super().__init__(tag)

        # Create a namespace for thermal transport equations

        thermal_transport = self.create_namespace('thermal_transport')
        # Add the the thermal conductance equation
        thermal_transport.add_equations([Thermal_Conductance_Equation(k=k)])

        self.register_items([side1, side2])

        thermal_transport.T1 = side1.thermal_capacitance.T
        thermal_transport.T2 = side2.thermal_capacitance.T
        side1.thermal_capacitance.P += thermal_transport.P1
        side2.thermal_capacitance.P += thermal_transport.P2




class Thermal_Conductor_old(ConnectorTwoWay):
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
        self.side1.thermal_transport.P += thermal_transport.P1

        # Create variables T and P in side 2 binding - this is so we now the item we later bind will have these variable
        self.side2.thermal_transport.create_variable(name='T')
        self.side2.thermal_transport.create_variable(name='P')

        # Map variables between binding and internal variables for side 2 - this is so the variables of the binding side 2 item will be updated based on operations in the equation of the item
        thermal_transport.T2 = self.side2.thermal_transport.T
        self.side2.thermal_transport.P += thermal_transport.P2


class ThermalCapacitancesSeries(Subsystem):
    def __init__(self, tag, Tinit=100, T0=25, num_nodes=10, k=1):
        super().__init__(tag)

        items = []


        #Create N heat capacitances and connect them.

        inlet_node = Thermal_Capacitance('node0', C=10000, T0=Tinit)
        items.append(inlet_node)
        prev_node = inlet_node
        for i in range(1,num_nodes):

            #Create thermal conductor
            node = Thermal_Capacitance('node' + str(i), C=100, T0=T0)
            # Connect the last node to the new node with a conductor
            #thermal_conductor = Thermal_Conductor('thermal_conductor' + str(i), k=k)
            thermal_conductor = Thermal_Conductor('thermal_conductor' + str(i), k=k, side1=prev_node, side2=node)
            #thermal_conductor.bind(side1=prev_node, side2=node)
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

    X = []
    Y = []
    Z = []

    num_nodes =[10,100,1000,2000,10000]
    Tinit = 100
    T0 = 25
    k = 1


    fig = []
    method_ns = 'LevenbergMarquardt'
    method_scipy = 'BDF'
    dt = {'numerous': [], 'scipy': []}
    for i in num_nodes:#range(1,num_nodes+1):
        fig.append(go.Figure())

        m = Model(ThermalCapacitancesSeries("tcs", num_nodes=i, Tinit=Tinit, T0=T0, k=k))

        # Define simulation
        s_ns = Simulation(m, t_start=0, t_stop=1000, num=1000, num_inner=1, method=method_ns)

        s_scipy = Simulation(m, t_start=0, t_stop=1000, num=1000, num_inner=1, method=method_scipy)
        #
        # solve simulation
        dt_ns = timeit(s_ns)
        dt_scipy = timeit(s_scipy)

        dt['numerous'].append(dt_ns)
        dt['scipy'].append(dt_scipy)
        df_ns = s_ns.model.historian_df
        df_scipy = s_scipy.model.historian_df
        ydatalabel = f'tcs.node{i-1}.thermal_capacitance.T'
        #ydatalabel = f'tcs.node1.thermal_transport.T'

        fig[-1].update_xaxes(title_text='Time(s)')
        fig[-1].update_yaxes(title_text=ydatalabel)
        fig[-1].add_trace(go.Scatter(x=df_ns['time'], y=df_ns[ydatalabel], name=f'numerous solver {method_ns}'))
        fig[-1].add_trace(go.Scatter(x=df_scipy['time'], y=df_scipy[ydatalabel], name=f'scipy solver {method_scipy}'))

    fig.append(go.Figure())
    fig[-1].update_xaxes(title_text='Number of nodes')
    fig[-1].update_yaxes(title_text='Simulation time(s)')
    fig[-1].add_trace(go.Scatter(x=num_nodes, y=dt['numerous'], name=f'numerous solver {method_ns}'))
    fig[-1].add_trace(go.Scatter(x=num_nodes, y=dt['scipy'], name=f'scipy solver {method_scipy}'))

    for f in fig:
        f.show()







