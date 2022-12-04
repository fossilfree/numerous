from enum import Enum
from .context_managers import _active_connections
from .interfaces import ConnectorInterface, ModuleConnectionsInterface
from .clonable_interfaces import ClassVarSpec, Clonable
from .module import ModuleSpec
from .variables import Variable
from .exceptions import AlreadyConnectedError

class Directions(Enum):
    GET = 0
    SET = 1

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


class Connection():
    """
        A connection is respresenting two connectors connected
    """

    side1: ConnectorInterface
    side2: ConnectorInterface
    map: dict


    def __init__(self, side1: ConnectorInterface, side2: ConnectorInterface, map: dict):
        
        super(Connection, self).__init__()
        # Remember, this is just sides - directions determined on channel level!
        """
            map: dictionary where the keys are channels on side1 and values are channels on side 2
        """

        self.side1 = side1
        self.side2 = side2
        self.map = map

        _active_connections.get_active_manager_context().register_connection(self)


class ModuleConnections(ModuleConnectionsInterface, Clonable):
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

    def register_connection(self, connection: ConnectorInterface):

        self.connections.append(connection)

    def set_host(self, host, attr):
        if self._host is None:
            self._host = host
            self._host_attr = attr
        else:
            raise ValueError('!')

    def clone(self):
        clone = super(ModuleConnections, self).clone()
        clone.connections = self.connections
        return clone

def create_connections():
    """
        helper method to create a Connection context manager to capture defined connections of the module where is used.
    """
    return ModuleConnections()

class Connector(ConnectorInterface, ClassVarSpec):

    _channel_directions: dict = {}
    _connection: Connection

    def __init__(self, **channels):
        super(Connector, self).__init__(class_var_type=(ModuleSpec, Variable), set_parent_on_references=False)
        self.set_references({k: c[0] for k, c in channels.items()})
        self._channel_directions = {k: c[1] for k, c in channels.items()}
        self._connection = None

    def get_channels(self):
        return self.get_references_of_type((Variable, ModuleSpec))

    @property
    def channels(self):
        return self.get_channels()

    @property
    def channel_directions(self):
        return self._channel_directions

    def connect(self, other: ConnectorInterface, map: dict = None):

        if self.get_connection():
            raise AlreadyConnectedError()

        if map is None:
            map = {key: key for key in other.channels.keys()}


        # Check if all channels exist and have same signal and opposite direction
        for other_key, self_key in map.items():
            assert isinstance(other.channels[other_key], (ModuleSpec,)) and isinstance(self.channels[self_key], (ModuleSpec,)) or \
                   isinstance(other.channels[other_key], (Variable,)) and isinstance(self.channels[self_key], (Variable,)), \
                f"Cannot connect channel different types {self.channels[self_key]} != {other.channels[other_key]}"

            if isinstance(self.channels[self_key], Variable):
                assert other.channels[other_key].signal == self.channels[
                    self_key].signal, f"Cannot connect channel with signal {other.channels[other_key].signal} to channel with signal {self.channels[self_key].signal}"

            assert other.channel_directions[other_key] != self.channel_directions[
                self_key], f"Cannot connect channel of same directions <{self_key}> to <{other_key}> with direction {self.channel_directions[self_key]}"

        self._connection = Connection(side1=self, side2=other, map=map)


    def get_connection(self):
        return self._connection

    @property
    def connection(self):
        return self.get_connection()
    def __lshift__(self, other):
        self.connect(other)
    def __rshift__(self, other):
        other.connect(self)

    def clone(self):
        clone = super(Connector, self).clone()
        clone._channel_directions = self._channel_directions
        clone._connection = self._connection
        return clone

    def connect_reversed(self, **kwargs):

        channels = {}
        map = {}
        for k, v in kwargs.items():
            channels[k] = get_value_for(v) if self.channel_directions[k] == Directions.SET else set_value_from(v)
            map[k] = k

        reverse_connector = Connector(
            **channels
        )

        self.connect(reverse_connector, map)

        return reverse_connector