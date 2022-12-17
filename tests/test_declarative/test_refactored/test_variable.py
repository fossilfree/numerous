from uuid import uuid4

from numerous.declarative.variables import Variable
from numerous.declarative.signal import Signal, PhysicalQuantities, Units
import pytest

@pytest.fixture
def variable():
    return Variable()

def test_clone(variable):

    val = 1.0
    variable.value = val
    variable.signal = Signal(physical_quantity=PhysicalQuantities.Temperature, unit=Units.C)
    instance = variable.instance({})

    assert instance.value == val
    assert instance.signal == variable.signal
