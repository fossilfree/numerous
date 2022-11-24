import pytest

from numerous.declarative.bus import *
from numerous.declarative.exceptions import *
from numerous.declarative.specification import ItemsSpec, Module, ScopeSpec, ModuleSpec
from numerous.declarative.variables import Parameter
from .mock_objects import TestModule


def test_create_connector():
    class Variables(ScopeSpec):
        var1 = Parameter(0)

    variables = Variables()

    connector = Connector("connector")

    connector.add_variable_set(variables.var1)


def test_add_connection():
    class Variables(ScopeSpec):
        var1 = Parameter(0)

    variables = Variables()

    connector = Connector("connector")

    connector.add_variable_set(variables.var1)

    class Variables2(ScopeSpec):
        var1 = Parameter(0)

    variables2 = Variables2()

    connector2 = Connector("connector2")

    connector2.add_variable_get(variables2.var1)

    with create_connections() as connections:
        connector >> connector2

    with pytest.raises(AlreadyConnectedError):
        with create_connections() as connections:
            connector << connector2


def test_connector_module():
    connector = Connector("connector")

    module_spec = ModuleSpec(TestModule)
    module_spec.set_host(object(), "mod")

    connector.add_module_get(module_spec)

    connector2 = Connector("connector2")

    module_spec2 = ModuleSpec(TestModule)
    module_spec2.set_host(object(), "mod")

    connector2.add_module_set(module_spec2)

    with create_connections() as connections:
        connector >> connector2

    with pytest.raises(AlreadyConnectedError):
        connector << connector2


def test_create_connector_assign():
    module = ModuleSpec(TestModule)
    connector = Connector("connector",
                          channel1=get_value_for(module)
                          )

    assert list(connector.channels.keys())[0] == "channel1"


class TestModuleWConnector(Module):
    """
    Class implementing a test module
    """

    tag: str = 'mod_w_conn'

    class Variables(ScopeSpec):
        var1 = Parameter(0)

    variables = Variables()

    class Items(ItemsSpec):
        mod: TestModule

    items = Items()

    connector = Connector(
        var1=get_value_for(variables.var1),
        mod=get_value_for(items.mod)
    )

    def __init__(self, tag=None):
        super(TestModuleWConnector, self).__init__(tag)
        # self.items.mod = TestModule("test")


class TestModuleWConnector2(Module):
    """
    Class implementing a test module
    """

    tag: str = 'mod_w_conn2'

    class Variables(ScopeSpec):
        var1 = Parameter(0)

    variables = Variables()

    class Items(ItemsSpec):
        mod2: TestModule

    items = Items()

    connector = Connector(
        var1=set_value_from(variables.var1),
        mod=set_value_from(items.mod2)
    )

    def __init__(self, tag=None):
        super(TestModuleWConnector2, self).__init__(tag)
        self.items.mod2 = TestModule("test2")
        ...


class TestModuleWConnectorInOtherModule(Module):
    class Items(ItemsSpec):
        mod_w_conn_1: TestModuleWConnector
        mod_w_conn_2: TestModuleWConnector2

    items = Items()
    with create_connections() as connections:
        items.mod_w_conn_1.connector >> items.mod_w_conn_2.connector

    def __init__(self, tag):
        super(TestModuleWConnectorInOtherModule, self).__init__(tag)
        self.items.mod_w_conn_1 = TestModuleWConnector('tmwc')
        self.items.mod_w_conn_2 = TestModuleWConnector2('tmwc2')
        ...


def test_module_with_connector():
    t = TestModuleWConnectorInOtherModule('test')
    t.finalize()
    ...
