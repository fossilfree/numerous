import pytest
from numerous.declarative import Module, ScopeSpec, Variable, Connector, set_value_from, get_value_for, ItemsSpec, create_connections
from numerous.declarative.generator import generate_system
from numerous.declarative.exceptions import ItemNotAssignedError
@pytest.fixture
def TestModule():
    class TestModule(Module):
        class Variables(ScopeSpec):
            a = Variable(0)

        variables = Variables()

        #connector = Connector(a=get_value_for(variables.a), optional=True)

    return TestModule

@pytest.fixture
def TestModuleMissingItem(TestModule):
    class TestModuleMissingItem(Module):
        class Items(ItemsSpec):
            mod_missing: TestModule

        items = Items()

    return TestModuleMissingItem

@pytest.fixture
def TestModuleConnection(TestModule):
    class TestModuleConnection(Module):
        class Items(ItemsSpec):
            mod_to: TestModule
            mod_from: TestModule

        items = Items()


        connector_to = Connector(var=get_value_for(items.mod_to.variables.a))#mod=get_value_for(items.mod_to), )
        connector_from = Connector(var=set_value_from(items.mod_from.variables.a))#mod=set_value_from(items.mod_from), )

        with create_connections() as connections:

            connector_to.connect(connector_from)
            ...

        def __init__(self):
            super(TestModuleConnection, self).__init__()

            self.items.mod_from = TestModule()
            self.items.mod_to = TestModule()

            print('var init: ', self.items.mod_to.module_spec.variables.a._id)

    return TestModuleConnection


def test_generator(TestModule):

    mod1 = TestModule()

    sys = generate_system("system", mod1)

def test_generator(TestModuleMissingItem):

    mod1 = TestModuleMissingItem()
    with pytest.raises(ItemNotAssignedError):

        sys = generate_system("system", mod1)

def test_module_connection(TestModuleConnection):
    mod = TestModuleConnection()
    sys = generate_system("system", mod)
