import pytest

from numerous.engine.variables import _VariableFactory, OverloadAction, VariableType, Variable
from numerous import VariableDescription
from .test_equations import *


@pytest.fixture
def constant():
    var_desc = VariableDescription(tag='test_derivative', initial_value=0,
                                   type=VariableType.CONSTANT,
                                   on_assign_overload=OverloadAction.SUM)
    return _VariableFactory._create_from_variable_desc_unbound(variable_description=var_desc, initial_value=0)


@pytest.fixture
def derivative():
    var_desc = VariableDescription(tag='test_constant', initial_value=0,
                                   type=VariableType.DERIVATIVE,
                                   on_assign_overload=OverloadAction.RaiseError)
    return _VariableFactory._create_from_variable_desc_unbound(variable_description=var_desc, initial_value=0)


def test_allow_update_false(derivative):
    x_dot = derivative
    x_dot.allow_update = False

    with pytest.raises(ValueError, match=r".*It is not possible to reassign variable.*"):
        x_dot.value = 1


def test_allow_update_true(derivative):
    x_dot = derivative
    x_dot.allow_update = True
    x_dot.value = 1
    assert x_dot.value == 1


def test_update_constant(constant):
    pi = constant
    with pytest.raises(ValueError, match=r".*not possible to reassign constant.*"):
        pi.value = 1


def test_error_on_nonnumeric_state(constant):
    with pytest.raises(ValueError, match=r".*State must be float or integer.*"):
        TestEq_dictState()
