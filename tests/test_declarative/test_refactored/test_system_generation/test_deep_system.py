from numerous.declarative import ItemsSpec, Module, ScopeSpec, Parameter, Connector, equation, create_connections
from numerous.declarative.variables import State, Parameter, Constant
from numerous.declarative.connector import get_value_for, set_value_from
from numerous.declarative.module import ModuleSpec
from numerous.declarative.debug_utils import print_all_variables
from numerous.declarative.generator import generate_system

import logging
import pytest

T_az = 273.15
Cp_W = 4190

@pytest.fixture
def ThermalMass():
    class ThermalMass(Module):
        class Variables(ScopeSpec):
            T, T_dot = State(0)
            P = Parameter(0)
            C = Constant(100 * 4200)

        variables = Variables()

        @equation(variables)
        def eq(self, scope: Variables):
            scope.T_dot = 0.001 * scope.P

        def __init__(self, T):
            super(ThermalMass, self).__init__()
            self.variables.T.value = T

    return ThermalMass

@pytest.fixture
def Conductor(ThermalMass):
    class Conductor(Module):
        """
            Model of a pipe using a fixed flow to propagate energy from one volume to another.
        """

        class Variables(ScopeSpec):
            """
            Variables for Fixed Flow Model
            """
            side1_P: Parameter = Parameter(0)
            side1_T: Parameter = Parameter(20)
            side2_P = Parameter(0)
            side2_T = Parameter(20)

        variables = Variables()

        class Items(ItemsSpec):
            """
            Items for fixed flow model
            """
            side1: ThermalMass
            side2: ThermalMass

        items = Items()

        # Map side1_ variables to side 1 item

        #This mapping will only be found in module spec on this module
        items.side1.variables.P += variables.side1_P
        #This mapping will be found in both module and module spec
        variables.side1_T = items.side1.variables.T

        # Map side2_ variables to side 2 item
        items.side2.variables.P += variables.side2_P
        variables.side2_T = items.side2.variables.T

        def __init__(self, side1: ThermalMass, side2: ThermalMass):
            super(Conductor, self).__init__()

            # Assign the side1 and side2 control volumes
            self.items.side1 = side1
            self.items.side2 = side2


        @equation(variables)
        def diff(self, scope: Variables):
            P = (scope.side1_T - scope.side2_T) * 100
            scope.side1_P = -P
            scope.side2_P = P

    return Conductor

@pytest.fixture
def ThermalReservoir():
    class ThermalReservoir(Module):
        class Variables(ScopeSpec):
            T = Parameter(25)
            T_inlet = Parameter(0)
            h = Constant(100)
            P = Parameter(0)

        variables = Variables()

        connector = Connector(
            T=get_value_for(variables.T_inlet),
            P=set_value_from(variables.P)
        )

        @equation(variables)
        def eq(self, scope: Variables):
            scope.P = (scope.T - scope.T_inlet) * scope.h

        def __init__(self, T=0):
            super(ThermalReservoir, self).__init__()
            self.variables.T.value = T

    return ThermalReservoir

@pytest.fixture
def ThermalRelaxation(ThermalMass, ThermalReservoir):
    class ThermalRelaxation(Module):
        class Items(ItemsSpec):
            reservoir: ThermalReservoir
            mass: ThermalMass

        items = Items()

        with create_connections() as connections:
            items.reservoir.connector.connect_reversed(
                T=items.mass.variables.T,
                P=items.mass.variables.P
            )

        def __init__(self, T=0):
            super(ThermalRelaxation, self).__init__()
            self.items.mass = ThermalMass(T=0)
            self.items.reservoir = ThermalReservoir(T=T)

    return ThermalRelaxation

@pytest.fixture
def ConnectedVolumes(ThermalMass, Conductor):
    class ConnectedVolumes(Module):
        class Items(ItemsSpec):
            tm1: ThermalMass
            tm2: ThermalMass
            conductor: Conductor

        items = Items()

        def __init__(self, T1=20, T2=20):
            super(ConnectedVolumes, self).__init__()

            self.items.tm1 = ThermalMass(T=T1)
            self.items.tm2 = ThermalMass(T=T2)
            self.items.conductor = Conductor(side1=self.items.tm1, side2=self.items.tm2)
            ...

    return ConnectedVolumes

@pytest.fixture
def ForwardedThermalRelaxation(ThermalRelaxation):
    class ForwardedThermalRelaxation(Module):
        class Items(ItemsSpec):
            tr: ThermalRelaxation

        items = Items()

        def __init__(self, T=0):
            super(ForwardedThermalRelaxation, self).__init__()

            self.items.tr = ThermalRelaxation(T=T)

    return ForwardedThermalRelaxation

@pytest.fixture
def ForwardReservoir(ThermalReservoir):
    class ForwardReservoir(Module):

        class Items(ItemsSpec):
            reservoir: ThermalReservoir

        items = Items()

        connector = items.reservoir.connector

        def __init__(self, T=0):
            super(ForwardReservoir, self).__init__()

            self.items.reservoir = ThermalReservoir(T)
    return ForwardReservoir

@pytest.fixture
def ConnectForwardedReservoirs(ThermalMass, ForwardReservoir):
    class ConnectForwardedReservoirs(Module):
        class Items(ItemsSpec):
            tm1: ThermalMass
            tm2: ThermalMass
            fr1: ForwardReservoir
            fr2: ForwardReservoir

        items = Items()
        with create_connections() as connections:
            items.fr1.connector.connect_reversed(
                T=items.tm1.variables.T,
                P=items.tm1.variables.P
            )

            items.fr2.connector.connect_reversed(
                T=items.tm2.variables.T,
                P=items.tm2.variables.P
            )

        def __init__(self, T0=15, T1=10, T2=20):
            super(ConnectForwardedReservoirs, self).__init__()

            self.items.tm1 = ThermalMass(T=T0)
            self.items.tm2 = ThermalMass(T=T0)
            self.items.fr1 = ForwardReservoir(T=T1)
            self.items.fr2 = ForwardReservoir(T=T2)

    return ConnectForwardedReservoirs

@pytest.fixture
def ForwardModuleWithReverseConnections(ConnectForwardedReservoirs):

    class ForwardModuleWithReverseConnections(Module):
        class Items(ItemsSpec):
            cfr: ConnectForwardedReservoirs

        items = Items()

        def __init__(self, T1=10, T2=20):
            super(ForwardModuleWithReverseConnections, self).__init__()

            self.items.cfr = ConnectForwardedReservoirs(T1=T1, T2=T2)
    return ForwardModuleWithReverseConnections


@pytest.fixture
def MultipleLinkedMasses(ThermalMass, Conductor):

    class MultipleLinkedMasses(Module):
        class Items(ItemsSpec):
            tm_right: ThermalMass
            tm_left: ThermalMass
            conductor_left: Conductor
            conductor_right: Conductor
            tm_middle: ThermalMass

        items = Items()

        def __init__(self, T_left, T_middle, T_right):
            super(MultipleLinkedMasses, self).__init__()

            self.items.tm_left = ThermalMass(T=T_left)
            #tm_middle = local("tm_middle", ThermalMass(T=10))
            self.items.tm_middle = ThermalMass(T=T_middle)
            self.items.conductor_left = Conductor(side1=self.items.tm_left, side2=self.items.tm_middle)

            self.items.tm_right = ThermalMass(T=T_right)

            self.items.conductor_right = Conductor(side1=self.items.tm_right, side2=self.items.tm_middle)
    return MultipleLinkedMasses

@pytest.fixture
def ForwardMultipleLinkedMasses(MultipleLinkedMasses):
    class ForwardMultipleLinkedMasses(Module):

        class Items(ItemsSpec):
            fmlm: MultipleLinkedMasses

        items = Items()

        def __init__(self, T_left=0, T_middle=0, T_right=0):
            super(ForwardMultipleLinkedMasses, self).__init__()

            self.items.fmlm = MultipleLinkedMasses(T_left=T_left, T_middle=T_middle, T_right=T_right)
    return ForwardMultipleLinkedMasses

def test_conductor_mappings(Conductor, ThermalMass):
    tm = ThermalMass(T=10)
    tm2 = ThermalMass(T=20)
    conductor  = Conductor(tm, tm2)
    ...

def test_connected_volumes(ConnectedVolumes):

    T1=20
    T2=10
    T= (T1+T2)/2
    test_system = ConnectedVolumes(T1=T1, T2=T2)



    system = generate_system('system', test_system)




    from numerous.engine import simulation
    from numerous.engine import model
    from matplotlib import pyplot as plt

    # Define simulation
    s = simulation.Simulation(model.Model(system, use_llvm=False), t_start=0, t_stop=500.0, num=1000, num_inner=100,
                              max_step=1)

    # Solve and plot
    s.solve()

    df = s.model.historian_df

    def last(var):
        return df[var.native_ref.path.primary_path].tail(1).values[0]

    # Run check that all WaterVol flow (F) are zero
    print_all_variables(test_system, df)
    # Cold side
    assert last(test_system.items.conductor.variables.side1_T) == pytest.approx(T)
    assert last(test_system.items.conductor.variables.side2_T) == pytest.approx(T)

def test_connectors(ThermalRelaxation):
    T=10
    test_system = ThermalRelaxation(T)

    system = generate_system('system', test_system)

    from numerous.engine import simulation
    from numerous.engine import model
    from matplotlib import pyplot as plt

    # Define simulation
    s = simulation.Simulation(model.Model(system, use_llvm=False), t_start=0, t_stop=500.0, num=1000, num_inner=100,
                              max_step=1)

    # Solve and plot
    s.solve()

    df = s.model.historian_df

    def last(var):
        return df[var.native_ref.path.primary_path].tail(1).values[0]

        # Run check that all WaterVol flow (F) are zero

    print_all_variables(test_system, df)
    # Cold side
    assert last(test_system.items.mass.variables.T) == pytest.approx(T)

def test_forwarded_reservoirs(ConnectForwardedReservoirs):
    T0 = 15
    T1 = 10
    T2 = 20

    test_system = ConnectForwardedReservoirs(T0, T1, T2)

    system = generate_system('system', test_system)

    from numerous.engine import simulation
    from numerous.engine import model
    from matplotlib import pyplot as plt

    # Define simulation
    s = simulation.Simulation(model.Model(system, use_llvm=False), t_start=0, t_stop=500.0, num=1000, num_inner=100,
                              max_step=1)

    # Solve and plot
    s.solve()

    df = s.model.historian_df

    def last(var):
        return df[var.native_ref.path.primary_path].tail(1).values[0]

        # Run check that all WaterVol flow (F) are zero

    print_all_variables(test_system, df)
    # Cold side
    assert last(test_system.items.tm1.variables.T) == pytest.approx(T1)
    assert last(test_system.items.tm2.variables.T) == pytest.approx(T2)

def test_fowarded_connector(ForwardedThermalRelaxation):
    T = 10
    test_system = ForwardedThermalRelaxation(T=T)

    system = generate_system('system', test_system)

    from numerous.engine import simulation
    from numerous.engine import model
    from matplotlib import pyplot as plt

    # Define simulation
    s = simulation.Simulation(model.Model(system, use_llvm=False), t_start=0, t_stop=500.0, num=1000, num_inner=100,
                              max_step=1)

    # Solve and plot
    s.solve()

    df = s.model.historian_df

    def last(var):
        return df[var.native_ref.path.primary_path].tail(1).values[0]

        # Run check that all WaterVol flow (F) are zero

    print_all_variables(test_system, df)
    # Cold side
    assert last(test_system.items.tr.items.mass.variables.T) == pytest.approx(T)

def test_forwarded_reverse_connectors(ForwardModuleWithReverseConnections):
    T1 = 10
    T2 = 20

    test_system = ForwardModuleWithReverseConnections(T1=T1, T2=T2)

    system = generate_system('system', test_system)

    from numerous.engine import simulation
    from numerous.engine import model
    from matplotlib import pyplot as plt

    # Define simulation
    s = simulation.Simulation(model.Model(system, use_llvm=False), t_start=0, t_stop=500.0, num=1000, num_inner=100,
                              max_step=1)

    # Solve and plot
    s.solve()

    df = s.model.historian_df

    def last(var):
        return df[var.native_ref.path.primary_path].tail(1).values[0]

        # Run check that all WaterVol flow (F) are zero

    print_all_variables(test_system, df)
    # Cold side
    assert last(test_system.items.cfr.items.tm1.variables.T) == pytest.approx(T1)
    assert last(test_system.items.cfr.items.tm2.variables.T) == pytest.approx(T2)

def test_multiple_linked(MultipleLinkedMasses):
    T_left=60
    T_right=T_middle=0
    T_mean=(T_left+T_right+T_middle)/3

    test_system = MultipleLinkedMasses(T_left=T_left, T_middle=T_middle, T_right=T_right)

    system = generate_system('system', test_system)

    from numerous.engine import simulation
    from numerous.engine import model
    from matplotlib import pyplot as plt

    # Define simulation
    s = simulation.Simulation(model.Model(system, use_llvm=False), t_start=0, t_stop=500.0, num=1000, num_inner=100,
                              max_step=1)

    # Solve and plot
    s.solve()

    df = s.model.historian_df

    def last(var):
        return df[var.native_ref.path.primary_path].tail(1).values[0]

        # Run check that all WaterVol flow (F) are zero

    print_all_variables(test_system, df)

    assert last(test_system.items.tm_left.variables.T) == pytest.approx(T_mean, abs=0.01)
    assert last(test_system.items.tm_middle.variables.T) == pytest.approx(T_mean, abs=0.01)
    assert last(test_system.items.tm_right.variables.T) == pytest.approx(T_mean, abs=0.01)

def test_forwarded_multiple_linked(ForwardMultipleLinkedMasses):
    T_left = 60
    T_right = T_middle = 0
    T_mean = (T_left + T_right + T_middle) / 3

    test_system = ForwardMultipleLinkedMasses(T_left=T_left, T_middle=T_middle, T_right=T_right)



    system = generate_system('system', test_system)



    from numerous.engine import simulation
    from numerous.engine import model
    from matplotlib import pyplot as plt

    # Define simulation
    s = simulation.Simulation(model.Model(system, use_llvm=False), t_start=0, t_stop=500.0, num=1000, num_inner=100,
                              max_step=1)

    # Solve and plot
    s.solve()

    df = s.model.historian_df

    def last(var):
        return df[var.native_ref.path.primary_path].tail(1).values[0]

        # Run check that all WaterVol flow (F) are zero

    print_all_variables(test_system, df)
    # Cold side
    assert last(test_system.items.fmlm.items.tm_left.variables.T) == pytest.approx(T_mean, abs=0.01)
    assert last(test_system.items.fmlm.items.tm_middle.variables.T) == pytest.approx(T_mean, abs=0.01)
    assert last(test_system.items.fmlm.items.tm_right.variables.T) == pytest.approx(T_mean, abs=0.01)


