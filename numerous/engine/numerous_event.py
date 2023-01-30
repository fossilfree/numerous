from typing import Optional, List, Callable, Dict
class NumerousEvent:
    key: str
    compiled: bool
    is_external: bool
    compiled_functions: Optional[Dict[str, Callable]]
    action: Callable

class StateEvent(NumerousEvent):

    def __init__(self, key, condition, action, compiled, terminal, direction, compiled_functions=None):
        self.key = key
        self.condition = condition
        self.action = action
        self.compiled = compiled
        self.compiled_functions = compiled_functions
        self.direction = direction
        self.terminal = terminal


class TimestampEvent(NumerousEvent):
    def __init__(self, key, action, timestamps: list=None, periodicity: float=None, compiled_functions=None,
                 is_external: bool=False):
        self.key = key
        self.action = action
        self.compiled_functions = compiled_functions

        assert not (periodicity and timestamps), "you cannot specify both timestamps and periodicity for a " \
                                                 "time-stamp event"

        self.timestamps = timestamps
        self.periodicity = periodicity
        self.is_external = is_external
