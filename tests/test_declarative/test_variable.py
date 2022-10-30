from numerous.declarative.variables import Variable
from numerous.engine.variables import VariableDescription
from uuid import uuid4
import pytest

def test_instance():

    var = Variable()


    i1 = var.instance(name="i1", id=str(uuid4()), host="")
    i2 = var.instance(name="i2", id=str(uuid4()), host="")

    assert i1 != i2

    i1.set_variable(VariableDescription("i1"))
    i2.set_variable(VariableDescription("i2"))

