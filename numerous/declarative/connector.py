from enum import Enum

from .instance import Class
from .interfaces import ConnectorInterface, VariableInterface, ModuleInterface, ModuleSpecInterface, ModuleConnectionsInterface
from .exceptions import AlreadyConnectedError
from .context_managers import _active_connections

class Directions(Enum):
    GET = 0
    SET = 1

def reversed_direction(direction: Directions):
    return Directions.SET if direction == Directions.GET else Directions.GET

def get_value_for(object):
    """
        Helper method to specify that the object will need to be read from the Connector/Bus
    """
    return object, Directions.GET


def set_value_from(object):
    """
        Helper method to specify that the object will need to be written to the Connector/Bus
    """
    return object, Directions.SET


class Connection(Class):
    """
        A connection is respresenting two connectors connected
    """

    side1: ConnectorInterface
    side2: ConnectorInterface
    map: dict[str,str]
    directions: dict[str:Directions]
    processed: bool = False

    def __init__(self, side1: ConnectorInterface, side2: ConnectorInterface, map: dict, directions: dict, init=True):
        super(Connection, self).__init__()
        # Remember, this is just sides - directions determined on channel level!
        """
            map: dictionary where the keys are channels on side1 and values are channels on side 2
        """
        if init:
            self.side1 = side1
            self.side2 = side2
            self.map = map
            self.directions = directions

            _active_connections.get_active_manager_context().register_connection(self)

    @property
    def channels(self):
        for k, v in self.map.items():
            yield k, self.side1.channels[k][0], v, self.side2.channels[v][0], self.directions[k]

    def _instance_recursive(self, context:dict):
        instance = Connection(None, None, None, None, False)
        instance.side1 = self.side1.instance(context)
        instance.side2 = self.side2.instance(context)
        instance.directions = self.directions
        instance.map = self.map

        return instance

class ModuleConnections(ModuleConnectionsInterface, Class):
    connections = []

    def __init__(self, *args):
        super(ModuleConnections, self).__init__()

        self._host = None
        self.connections = list(args)

    def __enter__(self):
        _active_connections.set_active_manager_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _active_connections.clear_active_manager_context(self)

    def register_connection(self, connection: Connection):

        self.connections.append(connection)

    def _instance_recursive(self, context:dict):
        instance = ModuleConnections(*(v.instance(context) for v in self.connections))

        return instance

def create_connections():
    """
        helper method to create a Connection context manager to capture defined connections of the module where is used.
    """
    return ModuleConnections()


class Connector(Class, ConnectorInterface):
    channels: dict[str:tuple[VariableInterface|ModuleInterface], Directions]
    connection: Connection
    optional: False

    def __init__(self, optional=False, **kwargs):
        super(Connector, self).__init__()
        self.channels = kwargs
        self.connection = None
        self.optional = optional


        for k, v in self.channels.items():
            if not isinstance(v, tuple):
                raise TypeError("Did you miss a call to get_value_for or set_value_from")
            setattr(self, k, v[0])

    def _instance_recursive(self, context):
        instance = Connector(**{k: (v[0].instance(context), v[1]) for k, v in self.channels.items()})
        #instance.connection = Connection(self.connection.side1.instance(context), self.connection.side2, self.connection.map) if self.connection else None
        instance.optional = self.optional
        return instance

    def connect(self, other: ConnectorInterface, map: dict = None):

        if self.connection:
            raise AlreadyConnectedError()

        if map is None:
            map = {key: key for key in other.channels.keys()}


        directions = {}

        # Check if all channels exist and have same signal and opposite direction
        for other_key, self_key in map.items():
            assert isinstance(other.channels[other_key][0], (ModuleInterface,ModuleSpecInterface)) and isinstance(self.channels[self_key][0], (ModuleInterface,ModuleSpecInterface)) or \
                   isinstance(other.channels[other_key][0], (VariableInterface,)) and isinstance(self.channels[self_key][0], (VariableInterface,)), \
                f"Cannot connect channel different types {self.channels[self_key][0]} != {other.channels[other_key][0]}"

            if isinstance(self.channels[self_key][0], VariableInterface):
                assert other.channels[other_key][0].signal == self.channels[
                    self_key][0].signal, f"Cannot connect channel with signal {other.channels[other_key][0].signal} to channel with signal {self.channels[self_key][0].signal}"

            assert other.channels[other_key][1] != self.channels[
                self_key][1], f"Cannot connect channel of same directions <{self_key}> to <{other_key}> with direction {self.channels[self_key][1]}"

            directions[other_key] = self.channels[self_key][1]

        self.connection = Connection(self, other, map, directions)

    def connect_reversed(self, **kwargs):
        reverse_connector = Connector(**{k: (v, reversed_direction(self.channels[k][1])) for k, v in kwargs.items()})
        self.connect(reverse_connector)

    @property
    def connected(self):
        return self.connection is not None
    def check_connection(self):
        if self.optional or self.connected:
            return
        raise ConnectionError("Connector is not connected!")

    def get_connection(self):
        return self.connection

    def __lshift__(self, other):
        self.connect(other)
    def __rshift__(self, other):
        other.connect(self)
