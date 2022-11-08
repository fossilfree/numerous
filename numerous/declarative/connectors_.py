from enum import Enum
import dataclasses
from typing import Any
from .specification import Module

class Directions(Enum):
    GET = 0
    SET = 1
    ADD = 2

class PhysicalQuantities(Enum):
    Flow = 0
    Temperature = 1


class Units(Enum):
    L_s = "L/s"
    C = "C"


@dataclasses.dataclass
class Signal:
    physical_quantity: PhysicalQuantities
    unit: Units
    variable: Any

    def __cmp__(self, other):
        if not isinstance(other, Signal):
            raise TypeError(f"Can only compare signal to antoher signal not to: {other.__class__}")
        return (self.physical_quantity == other.physical_quantity) and (self.unit == other.unit)

class Flow(Signal):

    physical_quantity = PhysicalQuantities.Flow
    unit = Units.L_s

@dataclasses.dataclass
class Connector:
    direction: Directions

@dataclasses.dataclass
class SignalConnector(Connector):

    signal: Signal

class ModuleConnector(Connector):
    module: Module.__class__


class MismatchingPhysicalQuantitiesError(Exception):
    ...

class MismatchingUnitsError(Exception):
    ...

class OnlyOneSetConnectorAllowed(Exception):
    ...

class OnlyAddConnectorsAllowed(Exception):
    ...

class OnlySetConnectorAllowed(Exception):
    ...

class NoGetConnectors(Exception):
    ...

class NoSetConnectors(Exception):
    ...

class Connection:

    def __init__(self):
        self.signal_physical_quantity = None
        self.signal_unit = None

        self.set_connector = None
        self.add_connectors = []
        self.get_connectors = []

    def connect_signals(*signal_connectors):

        signal_physical_quantity = None
        signal_unit = None

        set_connector = None
        add_connectors = []
        get_connectors = []


        for i, signal_connector in enumerate(signal_connectors):
            if i==0:
                signal_physical_quantity = signal_connector.signal.physical_quantity
                signal_unit = signal_connector.signal.unit

            if signal_physical_quantity != signal_connector.signal.physical_quantity:
                raise MismatchingPhysicalQuantitiesError(f"Cannot connect different physical quantities. Physical quantity already set {signal_physical_quantity.name}, trying to add {signal_connector.signal.physical_quantity.name}")

            if signal_unit != signal_connector.signal.unit:
                raise MismatchingUnitsError(
                    f"Cannot connect different units. Physical quantity already set {signal_unit.name}, trying to add {signal_connector.signal.unit.name}")

            if signal_connector.direction == Directions.SET:
                if set_connector is not None:
                    raise OnlyOneSetConnectorAllowed("You can only have one set connector in a connection.")
                if len(add_connectors)>0:
                    raise OnlyAddConnectorsAllowed("Since an ADD connector is already added you cannot add a SET connector")

                set_connector = signal_connector

            elif signal_connector.direction == Directions.GET:
                get_connectors.append(signal_connector)

            elif signal_connector.direction == Directions.ADD:
                if set_connector is not None:
                    raise OnlySetConnectorAllowed()

                add_connectors.append(signal_connector)

        if len(get_connectors) == 0:
            raise NoGetConnectors("A connection must have at lease one get connector!")

        if len(add_connectors) == 0 and set_connector is None:
            raise NoSetConnectors("A connection must have at lease one add connector or exactly one set connector!")

        for get_connector in get_connectors:


# physical quantity
# unit
# get/set
