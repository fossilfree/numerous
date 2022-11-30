import dataclasses
from enum import Enum


class PhysicalQuantities(Enum):
    Module = -1
    Default = 0
    Flow = 1
    Temperature = 2


class Units(Enum):
    NA = ""
    L_s = "L/s"
    C = "C"


@dataclasses.dataclass
class Signal:
    """
        Class to represent a physical signal specification.
    """
    _physical_quantity: PhysicalQuantities
    _unit: Units

    def __init__(self, physical_quantity: PhysicalQuantities, unit: Units):
        self.physical_quantity = physical_quantity
        self.unit = unit

    @property
    def physical_quantity(self):
        return self._physical_quantity

    @physical_quantity.setter
    def physical_quantity(self, physical_quantity:PhysicalQuantities):
        self._physical_quantity = physical_quantity

    @property
    def unit(self):
        return self._physical_quantity

    @unit.setter
    def unit(self, unit: Units):
        self._unit = unit

    def __cmp__(self, other):
        if not isinstance(other, Signal):
            raise TypeError(f"Can only compare signal to antoher signal not to: {other.__class__}")
        return (self.physical_quantity == other.physical_quantity) and (self.unit == other.unit)


default_signal = Signal(physical_quantity=PhysicalQuantities.Default, unit=Units.NA)
module_signal = Signal(physical_quantity=PhysicalQuantities.Module, unit=Units.NA)
