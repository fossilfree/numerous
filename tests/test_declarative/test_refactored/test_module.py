from numerous.declarative import ItemsSpec, Module, ModuleSpec, ScopeSpec, Parameter, Connector, EquationSpec
from numerous.declarative.connector import get_value_for, set_value_from


import pytest

@pytest.fixture
def TestModule():
    class TestModule(Module):

        class Items(ItemsSpec):
            a: Module

        items = Items()

        class Variables(ScopeSpec):

            var1 = Parameter(0)
            var2 = Parameter(0)

        variables = Variables()

        connector = Connector(
            var1=set_value_from(variables.var1)
        )

        connector2 = Connector(
            var1=get_value_for(variables.var2)
        )

        connector.connect(connector2)

        @EquationSpec(variables)
        def eq(self, scope:Variables):
            scope.var1 = 1.0


    return TestModule

def test_module(TestModule):

    test_module = TestModule()

    assert isinstance(test_module, Module)

    assert isinstance(test_module.items.a, ModuleSpec)
    assert TestModule.connector.get_connection() == TestModule.connector2

    assert TestModule.variables != test_module.variables
    assert TestModule.connector != test_module.connector


    assert TestModule.variables.var1 != test_module.variables.var1
    assert TestModule.variables.var1 != test_module.variables.var1

    assert TestModule.variables.var1 == TestModule.connector.var1


    assert test_module.variables.var1 == test_module.connector.var1

    assert test_module.connector.get_connection() == test_module.connector2


    assert test_module.variables.var1.get_path(test_module) == test_module.connector.var1.get_path(test_module)


