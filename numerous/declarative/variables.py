from enum import Enum
from .instance import Class
from .interfaces import VariableInterface
from .signal import Signal, PhysicalQuantities, Units
from .context_managers import _active_mappings

class MappingTypes(Enum):
    ASSIGN = 0
    ADD = 1

class Variable(Class, VariableInterface):

    _mappings:list[tuple[MappingTypes, VariableInterface]] = []
    signal: Signal
    value: float
    path: tuple[str] = None

    def __init__(self, value=None, signal: Signal = Signal(physical_quantity=PhysicalQuantities.Default, unit=Units.NA)):
        Class.__init__(self)
        self.signal = signal
        self.value = value

    def set_mappings(self, mappings:list):
        self._mappings = mappings

    def add_sum_mapping(self, mapping):
        self._mappings.append((MappingTypes.ADD, mapping))
        #_active_mappings.get_active_manager_context().add(self, mapping)

    def add_assign_mapping(self, mapping):
        self._mappings.append((MappingTypes.ASSIGN, mapping))
        #_active_mappings.get_active_manager_context().assign(self, mapping)

    def __add__(self, other):
        self.add_sum_mapping(other)
        return self

    def _instance_recursive(self, context):
        instance = self.__class__()
        instance._context = context
        instance.set_mappings([(v[0], v[1].instance(context)) for v in self._mappings])
        instance.value = self.value
        instance.signal = self.signal
        return instance

    @property
    def mappings(self):
        return self._mappings


class Parameter(Variable):
    """
    Declaration of a Parameter
    """
    ...


class Constant(Variable):
    """
    Declaration of a Constant. A constant cannot be changed.
    """


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
    return var, Variable(value=0)



