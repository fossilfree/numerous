import pytest

from numerous.engine.system import ConnectorTwoWay, Item
from numerous import VariableDescription, VariableType


@pytest.fixture
def simple_item_1():
    class TestItem(Item):
        def __init__(self, tag):
            super().__init__(tag)
            test_namespace = self.create_namespace('test_namespace')
            var_desc = VariableDescription(tag='T', initial_value=0,
                                           type=VariableType.PARAMETER)
            test_namespace.create_variable_from_desc(var_desc)
            var_desc = VariableDescription(tag='T1', initial_value=0,
                                           type=VariableType.PARAMETER)
            var_desc = VariableDescription(tag='P', initial_value=0,
                                           type=VariableType.PARAMETER)
            test_namespace.create_variable_from_desc(var_desc)

    return TestItem('test_item')


@pytest.fixture
def simple_item_2():
    class TestItem(Item):
        def __init__(self, tag):
            super().__init__(tag)
            test_namespace = self.create_namespace('test_namespace')
            var_desc = VariableDescription(tag='T', initial_value=11,
                                           type=VariableType.PARAMETER)
            test_namespace.create_variable_from_desc(var_desc)
            var_desc = VariableDescription(tag='T1', initial_value=11,
                                           type=VariableType.PARAMETER)
            test_namespace.create_variable_from_desc(var_desc)

    return TestItem('test_item')


@pytest.fixture
def two_way_connector_item():
    class TestConductor(ConnectorTwoWay):
        def __init__(self, tag):
            super(TestConductor, self).__init__(tag)
            self.create_namespace('test_namespace')
            self.side1.test_namespace.create_variable(name='T')
            self.side1.test_namespace.create_variable(name='P')
            var_desc = VariableDescription(tag='P1', initial_value=11,
                                           type=VariableType.PARAMETER)
            self.test_namespace.create_variable_from_desc(var_desc)
            self.side2.test_namespace.create_variable(name='T')

    return TestConductor('test_connector')


def test_binding_multiple_add(two_way_connector_item, simple_item_1):
    with pytest.raises(ValueError, match=r".*already binded to binding.*"):
        two_way_connector_item.bind(side1=simple_item_1)
        two_way_connector_item.bind(side1=simple_item_1)


def test_binding_1(two_way_connector_item, simple_item_1):
    assert two_way_connector_item.side1.test_namespace.T.value is None
    two_way_connector_item.bind(side1=simple_item_1)

    assert two_way_connector_item.side1.test_namespace.T.value == 0
    simple_item_1.test_namespace.T.value = 10

    assert two_way_connector_item.side1.test_namespace.T.value == 10
    two_way_connector_item.side1.test_namespace.T.value = 1

    assert simple_item_1.test_namespace.T.value == 1



def test_binding_2(two_way_connector_item, simple_item_1):
    assert two_way_connector_item.side1.test_namespace.T.value is None
    two_way_connector_item.side1.test_namespace.P = two_way_connector_item.test_namespace.P1
    assert simple_item_1.test_namespace.P.value == 0
    two_way_connector_item.bind(side1=simple_item_1)

    assert simple_item_1.test_namespace.P.value == 11
    two_way_connector_item.test_namespace.P1.value = 10
    assert simple_item_1.test_namespace.P.value == 10


def test_binding_3(two_way_connector_item, simple_item_1, simple_item_2):
    assert two_way_connector_item.side1.test_namespace.T.value is None
    assert two_way_connector_item.side2.test_namespace.T.value is None
    two_way_connector_item.bind(side1=simple_item_1, side2=simple_item_2)

    assert two_way_connector_item.side1.test_namespace.T.value == 0
    simple_item_1.test_namespace.T.value = 10

    assert two_way_connector_item.side1.test_namespace.T.value == 10
    two_way_connector_item.side1.test_namespace.T.value = 1

    assert simple_item_1.test_namespace.T.value == 1

    assert two_way_connector_item.side2.test_namespace.T.value == 11
    simple_item_2.test_namespace.T.value = 10

    assert two_way_connector_item.side2.test_namespace.T.value == 10
    two_way_connector_item.side2.test_namespace.T.value = 1

    assert simple_item_2.test_namespace.T.value == 1
