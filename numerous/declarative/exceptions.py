from __future__ import annotations


class ChannelAlreadyOnBusError(Exception):
    ...


class AlreadyConnectedError(Exception):
    ...


class ChannelNotFound(Exception):
    ...


class WrongMappingDirectionError(Exception):
    ...


class MappingOutsideMappingContextError(Exception):
    ...


class ItemNotAssignedError(Exception):
    ...


class NoSuchItemInSpec(Exception):
    ...


class UnrecognizedItemSpec(Exception):
    ...


class MappingFailed(Exception):
    ...


class MappedToFixedError(Exception):
    ...


class NotMappedError(Exception):
    ...


class FixedMappedError(Exception):
    ...


class HostNotFoundError(Exception):
    ...


class DuplicateItemError(Exception):
    ...

class ItemAlreadyAssigned(Exception):
    ...

class ItemUndeclaredWarning(Warning):
    ...