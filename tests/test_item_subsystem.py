import pytest

from numerous.engine.system import Connector, Item, namespace
from numerous import VariableDescription, VariableType



@pytest.fixture
def item_with_namespace():

    item = Item('item_with_namespace')
    test_namespace = item.create_namespace('test_namespace')

    var_desc = VariableDescription(tag='A_parameter', initial_value=0, type=VariableType.PARAMETER)
    test_namespace.create_variable_from_desc(var_desc)

    var_desc = VariableDescription(tag='B_state', initial_value=0, type=VariableType.STATE)
    test_namespace.create_variable_from_desc(var_desc)

    return item


def test_add_namespace_twice(item_with_namespace):
    with pytest.raises(ValueError, match=r".*is already registered in item.*"):
        item_with_namespace.create_namespace('test_namespace')


def test_add_binding_twice(item_with_namespace):
    c = Connector('test')
    with pytest.raises(ValueError, match=r".*is already registered in connector.*"):
        c.create_binding('b1')
        c.create_binding('b1')


def test_add_mapping(item_with_namespace):
    ### It is not possible to use local variables for variables in namespace
    ### A_p = item_with_namespace.test_namespace.A_parameter
    ###
    test_namespace = item_with_namespace.test_namespace

    test_namespace.A_parameter.value = 1
    test_namespace.B_state.value = 0

    assert test_namespace.A_parameter.value != test_namespace.B_state.value

    test_namespace.A_parameter = test_namespace.B_state

    assert test_namespace.A_parameter.get_value() == 0
    assert bool(test_namespace.A_parameter.mapping) == 1
    assert test_namespace.B_state == test_namespace.A_parameter.mapping

    test_namespace.B_state.value = 10
    assert test_namespace.A_parameter.get_value() == 10

    test_namespace.A_parameter.value = 20
    assert test_namespace.B_state.get_value() == 10


def test_item_without_namespace():

    item = Item('item_without_namespace')

    default_namespace = item.default
    assert default_namespace

    with pytest.raises(AttributeError, match=r".*object has no attribute '.*'"):
        item.t1
        item.not_implemented_method

    item.create_namespace('t1')
    assert item.t1

    with pytest.raises(ValueError, match=r".*is already registered in item item_without_namespace"):
        item.create_namespace('default')
        item.create_namespace('t1')