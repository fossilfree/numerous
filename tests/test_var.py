import pytest

from numerous.engine.variables import _VariableFactory, OverloadAction, VariableType, Variable
from numerous import VariableDescription
from tests.test_equations import *

@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    import shutil
    shutil.rmtree('./tmp', ignore_errors=True)
    yield


@pytest.fixture
def constant():
    var_desc = VariableDescription(tag='test_derivative', initial_value=0,
                                   type=VariableType.CONSTANT)
    return _VariableFactory._create_from_variable_desc_unbound(variable_description=var_desc, initial_value=0)


@pytest.fixture
def derivative():
    var_desc = VariableDescription(tag='test_constant', initial_value=0,
                                   type=VariableType.DERIVATIVE)
    return _VariableFactory._create_from_variable_desc_unbound(variable_description=var_desc, initial_value=0)


def test_error_on_nonnumeric_state(constant):
    with pytest.raises(ValueError, match=r".*State must be float or integer.*"):
        TestEq_dictState()
