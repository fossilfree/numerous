import pytest
from numerous.engine.model import Model
from numerous.engine.system import Subsystem
from numerous.multiphysics import Equation, EquationBase


class StringVariables(Subsystem, EquationBase):
    def __init__(self, tag='system'):
        super(StringVariables, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('var1', 'not implemented')
        self.t1.add_equations([self])


class FloatVariables(Subsystem, EquationBase):
    def __init__(self, tag='system'):
        super(FloatVariables, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_parameter('var1', 1.0)
        self.t1.add_equations([self])


def test_throw_error():

    with pytest.raises(ValueError, match=r"Only numeric values allowed in variables*"):
        StringVariables()

    with pytest.raises(ValueError, match=r"Only numeric values allowed in variables*"):
        sys = FloatVariables()
        sys.t1.var1.value = "test"


