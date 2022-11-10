import dataclasses
from enum import Enum
from typing import Any


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
    physical_quantity: PhysicalQuantities
    unit: Units

    def __cmp__(self, other):
        if not isinstance(other, Signal):
            raise TypeError(f"Can only compare signal to antoher signal not to: {other.__class__}")
        return (self.physical_quantity == other.physical_quantity) and (self.unit == other.unit)

default_signal = Signal(physical_quantity=PhysicalQuantities.Default, unit=Units.NA)
module_signal = Signal(physical_quantity=PhysicalQuantities.Module, unit=Units.NA)