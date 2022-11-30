from enum import Enum
from .clonable_interfaces import Clonable, ParentReference, get_class_vars
from .signal import Signal, PhysicalQuantities, Units

class Operations(Enum):
    ADD = 1
    SUB = 2
    DIV = 3
    MUL = 4
    POW = 5
    FUNC = 6
    NEG = 7
    LT = 8
    GT = 9
    GET = 10
    LET = 11
    EQ = 12

class Variable(Clonable):
    _mappings: list
    _value: float
    signal: Signal
    _value: float

    def __init__(self, value=None, signal: Signal = Signal(physical_quantity=PhysicalQuantities.Default, unit=Units.NA)):

        super(Variable, self).__init__(clone_refs=False)

        self.signal = signal

        self._value = value


    def __add__(self, other):
        self.add_reference(other._id, other)

        return self

    def assign(self, other):
        self.add_reference(other._id, other)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def mappings(self):
        return self._references


    def clone(self):
        clone = super(Variable, self).clone()
        clone.value = self.value
        clone.signal = self.signal

        return clone

class Parameter(Variable):
    """
    Declaration of a Parameter
    """
    ...


class Constant(Variable):
    """
    Declaration of a Constant. A constant cannot be changed.
    """

    ...

class StateVar(Variable):
    """
    Declaration of a Constant. A constant cannot be changed.
    """

    ...

class Derivative(Variable):
    """
    Declaration of a Constant. A constant cannot be changed.
    """

    ...


def State(value):
    """
    Declaration of a State. States have time derivatives which are integrated to the state values of the system.
    """
    return StateVar(value=value), Derivative(value=0)


def integrate(var, integration_spec):
    var.integrate = integration_spec
    return var, Variable(value=0, construct=False)