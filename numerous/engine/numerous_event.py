import numpy as np
from typing import Optional, Callable, Dict, List
class NumerousEvent:
    """
    Base event for numerous engine
    """
    key: str
    compiled: bool
    is_external: bool
    compiled_functions: Optional[Dict[str, Callable]]
    action: Callable
    parent_path: Optional[List[str]]

class StateEvent(NumerousEvent):
    """
    A variant of :class:`~engine.numerous_event.NumerousEvent` which is used to generate state events. This is used
    internally in numerous engine.
    """

    def __init__(self, key: str, condition: Callable, action: Callable, direction: int, compiled: bool = False,
                 terminal: bool = False, compiled_functions = None,
                 is_external: bool = False, parent_path: Optional[List[str]] = None):
        """

        :param key: an identifier
        :param condition: a callable condition function with signature (t, variables) used for root finding to check if
        state event is triggered
        :param action: a callable action function with signature (t, variables) called once state event is triggered
        :param direction: a value that is either positive or negative depending on the direction of the event trigger
        :param compiled: an internal parameter
        :param terminal: an internal parameter
        :param compiled_functions: an internal parameter
        :param is_external: a bool, which determines if the action function is external or not (and will be compiled or
        not). This allows the user to create custom action functions that are not necessarily numba compilable.
        :param parent_path: internal parameter related to the item path, for which the event belongs to.
        """
        self.key = key
        self.condition = condition
        self.action = action
        self.compiled = compiled
        self.compiled_functions = compiled_functions
        self.direction = direction
        self.terminal = terminal
        self.is_external = is_external
        self.parent_path = parent_path


class TimestampEvent(NumerousEvent):
    """
    A variant of :class:`~engine.numerous_event.NumerousEvent` which is used to generate time stamp events. This is used
    internally in numerous engine.
    """
    def __init__(self, key: str, action: Callable, timestamps: list = None, periodicity: float = None,
                 compiled_functions=None, is_external: bool=False, parent_path: list[str] = None):
        """
        initialization function for time stamped events

        :param key: an identifier
        :param action: a callable action function with signature (t, variables)
        :param timestamps: an optional list of timestamps given as a list or np array
        :type timestamps: Optional[List, :class:`numpy.ndarray`]
        :param periodicity: an optional parameter that is used instead of timestamps when using a fixed periodicity for
        calling the action function.
        :param compiled_functions: internal parameter
        :param is_external: a bool, which determines if the action function is external or not (and will be compiled or
        not). This allows the user to create custom action functions that are not necessarily numba compilable.
        :param parent_path: internal parameter related to the item path, for which the event belongs to.
        """
        self.key = key
        self.action = action
        self.compiled_functions = compiled_functions

        assert not (periodicity and timestamps), "you cannot specify both timestamps and periodicity for a " \
                                                 "time-stamp event"

        assert not (not timestamps and not periodicity), "you must specify at least timestamps or periodicity"
        assert isinstance(timestamps, list) or isinstance(timestamps, np.ndarray) or not timestamps, \
            "timestamps must be list or np array"

        self.timestamps = np.array(timestamps) if isinstance(timestamps, list) else timestamps
        self.periodicity = periodicity
        self.is_external = is_external
        self.parent_path = parent_path
