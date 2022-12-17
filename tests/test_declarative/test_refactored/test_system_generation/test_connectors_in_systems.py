import pytest
from numerous.declarative import ItemsSpec, Module, ScopeSpec, Parameter, Connector, equation
from numerous.declarative.module import ModuleSpec
from numerous.declarative.variables import State, Parameter, Constant
from numerous.declarative.connector import get_value_for, set_value_from
from numerous.declarative.debug_utils import print_all_variables

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
def ThermalMassConnector(ThermalMass):

    class ThermalMassConnector(Module):

        class Items(ItemsSpec):
            tm: ThermalMass

        items = Items()

        connector = Connector(tm=get_value_for(items.tm))

    return ThermalMassConnector

@pytest.fixture
def ThermalMassConnectionAssign(ThermalMass, ThermalMassConnector):

    class ThermalMassConnectionAssign(Module):
        class Items(ItemsSpec):
            in_tm: ThermalMass
            tm_conn: ThermalMassConnector

        items = Items()

        connector = Connector(tm=set_value_from(items.in_tm))

        connector >> items.tm_conn.connector

        def __init__(self):
            super(ThermalMassConnectionAssign, self).__init__()

            self.items.in_tm = ThermalMass(T=20)
            self.items.tm_conn = ThermalMassConnector()

    return ThermalMassConnectionAssign

def test_assign_to_connected_module(ThermalMassConnectionAssign):

    test_system = ThermalMassConnectionAssign()