from enum import Enum
from .interfaces import ConnectorInterface
from .clonable_interfaces import ClassVarSpec
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


class Connector(ConnectorInterface, ClassVarSpec):

    _channel_directions: dict = {}


    def __init__(self, **channels):
        super(Connector, self).__init__(class_var_type=(ModuleSpec, Variable), set_parent_on_references=False)
        self.set_references({k: c[0] for k, c in channels.items()})
        self._channel_directions = {k: c[1] for k, c in channels.items()}

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
        for from_key, to_key in map.items():
            assert isinstance(other.channels[to_key], (ModuleSpec,)) and isinstance(self.channels[from_key], (ModuleSpec,)) or \
                   isinstance(other.channels[to_key], (Variable,)) and isinstance(self.channels[from_key], (Variable,)), \
                f"Cannot connect channel different types {self.channels[from_key]} != {other.channels[to_key]}"

            assert other.channels[to_key].signal == self.channels[
                from_key].signal, f"Cannot connect channel with signal {other.channels[to_key].signal} to channel with signal {self.channels[from_key].signal}"

            assert other.channel_directions[to_key] != self.channel_directions[
                from_key], f"Cannot connect channel of same directions <{to_key}> to <{from_key}> with direction {self.channel_directions[from_key]}"

        self.add_reference('_connection', other)

    def get_connection(self):
        connection = list(self.get_references_of_type((Connector,)).values())
        if len(connection)>0:
            return connection[0]

    def clone(self):
        clone = super(Connector, self).clone()
        clone._channel_directions = self._channel_directions
        return clone