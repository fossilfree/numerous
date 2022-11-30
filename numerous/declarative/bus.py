import dataclasses
from abc import ABC
from enum import Enum

from numerous.declarative.context_managers import _active_connections
from numerous.declarative.exceptions import ChannelAlreadyOnBusError, AlreadyConnectedError, ChannelNotFound, \
    WrongMappingDirectionError
from numerous.declarative.module import ModuleSpecInterface
from numerous.declarative.signal import Signal, module_signal
from numerous.declarative.utils import recursive_get_attr
from numerous.declarative.variables import Variable

from numerous.declarative.clonable_interfaces import ClassVarSpec

class Directions(Enum):
    GET = 0
    SET = 1


@dataclasses.dataclass
class Channel:
    """
        A channel represents a signal with a given name.
    """
    name: str
    signal: Signal


class BusInterface(ABC):
    name: str
    channels: {}
    connections: []

class Connection:
    """
        A connection is respresenting two connectors connected
    """

    side1: BusInterface
    side2: BusInterface
    map: dict
    _finalized = False

    def __init__(self, side1: BusInterface, side2: BusInterface, map: dict):
        # Remember, this is just sides - directions determined on channel level!
        """
            map: dictionary where the keys are channels on side1 and values are channels on side 2
        """
        self.side1 = side1
        self.side2 = side2
        self.map = map

        _active_connections.get_active_manager_context().register_connection(self)

    def finalize(self, host):
        self._finalized = True

        for from_attr, to_attr in self.map.items():
            print("connection: ", to_attr, " = ", from_attr)

            #self.side1.relativize(host)
            #self.side2.relativize(host)

            if not self.side1._relativized:
                raise ValueError("should be relativized")

            #print(self.side1._relativized)
            #print(self.side2._relativized)


            if self.side1.variable_mappings[from_attr][0][1] == Directions.SET:

                from_rel_path = self.side1.get_path(host)
                from_path = from_rel_path + self.side1.variable_mappings[from_attr][0][2]
                to_path = self.side2.get_path(host) + self.side2.variable_mappings[to_attr][0][2]
            else:
                from_rel_path = self.side2.get_path(host)

                from_path = from_rel_path + self.side2.variable_mappings[from_attr][0][2]
                to_path = self.side1.get_path(host) + self.side1.variable_mappings[to_attr][0][2]
            print(from_rel_path)
            #from_path = from_decl_var.get_path(host)
            print(from_path)

            from_var = recursive_get_attr(host, from_path)

            #to_path = to_decl_var.get_path(host)
            to_var = recursive_get_attr(host, to_path)
            if isinstance(to_var, ModuleSpecInterface):
                to_var._assigned_to = from_var
                ...
            else:
                #if to_var.additive:
                to_var.add_sum_mapping(from_var)
                #else:
                #    to_var.add_mapping(from_var)


class ModuleConnections:
    connections = []
    _host: str
    _host_attr: str

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

    def set_host(self, host, attr):
        if self._host is None:
            self._host = host
            self._host_attr = attr
        else:
            raise ValueError('!')

    def clone(self):
        clone = self.__class__(*self.connections.copy())
        clone._host = self._host
        clone._host_attr = self._host_attr
        return clone

    def finalize(self):
        for connection in self.connections:
            connection.finalize(self._host)


def create_connections():
    """
        helper method to create a Connection context manager to capture defined connections of the module where is used.
    """
    return ModuleConnections()


class Bus(BusInterface, ClassVarSpec):
    """
        A bus is a collection of channels to which connectors can connect to read/write variables on the bus.
    """

    _host: str
    _host_attr: str

    def __init__(self, name: str, parent=None, references=None, clone_refs=False):

        super(Bus, self).__init__(parent=parent, references=references, clone_refs=clone_refs)

        self.name = name
        self.channels = {}
        self.connections = []

    def set_host(self, host, attr):
        self._host = host
        self._host_attr = attr

    def get_path(self, parent):

        #_attr = self._host_attr

        if self == parent:

            path = []
            return path
        else:

            path = self._host.get_path(parent)
            return path

    def add_channel(self, channel: Channel):

        if channel.name in self.channels:
            raise ChannelAlreadyOnBusError()
        self.channels[channel.name] = channel

    def add_connection(self, other: BusInterface, map: dict = None):

        if other in self.connections:
            raise AlreadyConnectedError()

        if map is None:
            map = {channel.name: channel for channel in other.channels.values()}

        # Check if all channels exist and have same signal
        for other_channel, own_channel_name in map.items():
            if not own_channel_name in self.channels:
                self.add_channel(other_channel)

            assert other_channel.signal == self.channels[
                own_channel_name].signal, f"Cannot connect channel with signal {other_channel.signal} to channel with signal {self.channels[own_channel_name].signal}"

        self.connections[other] = map

    def finalize(self):
        raise NotImplementedError()


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


class Connector(Bus):
    """
        A connector is a Bus where the channels are linked to variables. This is to terminate the bus by reading/writing to actual variables.
        Connectors are used to define interfaces for a module.
    """
    variable_mappings = {}
    _relativized = False

    def __init__(self, name: str = None, **channels):
        super(Connector, self).__init__(name)
        self.variable_mappings = {}

        for channel_name, object_direction in channels.items():
            try:
                object, direction = object_direction

                if isinstance(object, ModuleSpecInterface):

                    self._add_module(object, channel_name, direction)
                elif isinstance(object, Variable):
                    self.add_variable(object, channel_name, direction)
            except TypeError:
                raise ValueError(
                    f"Could not unpack the channel <{channel_name}> into a variable and a direction. Have you forgot to wrap the variable or module with a <set_value_from> or <get_value_for> call?")

    def add_variable(self, variable: Variable, channel_name=None, direction=Directions.GET):
        if channel_name is None:
            channel_name = variable.name

        if not channel_name in self.variable_mappings:
            self.variable_mappings[channel_name] = []

        if not channel_name in self.channels:
            self.channels[channel_name] = Channel(name=channel_name, signal=variable.signal)

        self.variable_mappings[channel_name].append((variable, direction))

    def add_variable_set(self, variable, channel_name=None):
        self.add_variable(variable, channel_name, direction=Directions.SET)

    def add_variable_get(self, variable, channel_name=None):
        self.add_variable(variable, channel_name, direction=Directions.GET)

    def add_variable_add(self, variable, channel_name=None):
        self.add_variable(variable, channel_name, direction=Directions.ADD)

    def _check_not_connected(self, other):

        for connection in self.connections:
            if connection.side1 == other or connection.side2 == other:
                raise AlreadyConnectedError()
        # if len(self.connections)>0:
        #    raise AlreadyConnectedError()

    def _add_module(self, module: ModuleSpecInterface, channel_name: str, direction: Directions):

        if channel_name is None:
            channel_name = module._host_attr

        if not channel_name in self.variable_mappings:
            self.variable_mappings[channel_name] = []

        if not channel_name in self.channels:
            self.channels[channel_name] = Channel(name=channel_name, signal=module_signal)

        self.variable_mappings[channel_name].append((module, direction))

    def add_module_get(self, module: ModuleSpecInterface, channel_name: str = None):
        self._add_module(module, channel_name, Directions.GET)

    def add_module_set(self, module: ModuleSpecInterface, channel_name: str = None):
        self._add_module(module, channel_name, Directions.SET)

    def add_connection(self, other: BusInterface, map: dict = None, connection=None):

        self._check_not_connected(other)

        if map is None:
            map = {channel.name: channel.name for channel in other.channels.values()}

        # Check if all channels exist and have same signal
        for other_channel_name, own_channel_name in map.items():
            if not own_channel_name in self.channels:
                raise ChannelNotFound()

            other_channel: Channel = other.channels[other_channel_name]
            own_channel: Channel = self.channels[own_channel_name]

            if self.variable_mappings[own_channel_name][0][1] == Directions.GET:
                if isinstance(other, Connector) and not other.variable_mappings[other_channel_name][0][
                                                            1] == Directions.SET:
                    raise WrongMappingDirectionError()
            else:
                if isinstance(other, Connector) and not other.variable_mappings[other_channel_name][0][
                                                            1] == Directions.GET:
                    raise WrongMappingDirectionError()

            assert other.channels[other_channel_name].signal == self.channels[
                own_channel.name].signal, f"Cannot connect channel with signal {other_channel_name.signal} to channel with signal {self.channels[own_channel_name].signal}"

        if connection is None:
            connection = Connection(self, other, map)
            other.add_connection(self, map, connection)
        self.connections.append(connection)

        return connection

    #def clone(self):
    #    clone = self.__class__()
    #    clone.name = self.name
    #    clone.channels = self.channels.copy()
    #    clone.connections = self.connections.copy()
    #    clone.variable_mappings = self.variable_mappings.copy()
    #    clone._relativized = self._relativized
        #clone.variable_mappings = {k: [(vi[0].clone(), vi[1]) if not isinstance(vi[0], Variable) else (vi[0], vi[1]) for vi in v] for k, v in self.variable_mappings.items()}
    #    return clone

    def __rshift__(self, other: Bus):
        connection = self.add_connection(other)
        return connection

    def __lshift__(self, other: Bus):
        connection = self.add_connection(other)
        return connection

    # def __eq__(self, other):
    #    connection = self.add_connection(other)
    #    return connection

    def finalize(self):

        if len(self.connections) <= 0:
            raise ValueError(f"{self.name} is not connected!")

        for connection in self.connections:
            connection.finalize()

    def connect_reversed(self, **variables):
        reverse_connector = Connector(self.name)

        for name, channel in self.channels.items():
            reverse_connector.add_channel(Channel(name, channel.signal))
            print(self.variable_mappings[name])
            direction = self.variable_mappings[name][0][1]
            reverse_direction = Directions.SET if direction == Directions.GET else Directions.GET

            reverse_connector.variable_mappings[name] = [(variables[name], reverse_direction)]

        self.add_connection(reverse_connector)


    def relativize(self, host):
        if not self._relativized:
            relative_variable_mappings = {}
            for key, variable_mappings in self.variable_mappings.items():
                relative_mappings = []
                for mapping, direction in variable_mappings:
                    print(mapping)
                    if isinstance(mapping, list):
                        rel_map = mapping
                    else:
                        rel_map = mapping.get_path(host)
                    print('rel_map: ', rel_map)
                    relative_mappings.append((mapping, direction, rel_map))
                relative_variable_mappings[key] = relative_mappings
            self.variable_mappings = relative_variable_mappings

            self._relativized = True
