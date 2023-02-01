from typing import Optional, Callable, Dict
class NumerousEvent:
    key: str
    compiled: bool
    is_external: bool
    compiled_functions: Optional[Dict[str, Callable]]
    action: Callable
    parent_path: str

class StateEvent(NumerousEvent):

    def __init__(self, key, condition, action, compiled, terminal, direction, compiled_functions=None,
                 is_external=False, parent_path=None):
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
    def __init__(self, key, action, timestamps: list=None, periodicity: float=None, compiled_functions=None,
                 is_external: bool=False, parent_path=None):
        self.key = key
        self.action = action
        self.compiled_functions = compiled_functions

        assert not (periodicity and timestamps), "you cannot specify both timestamps and periodicity for a " \
                                                 "time-stamp event"

        self.timestamps = timestamps
        self.periodicity = periodicity
        self.is_external = is_external
        self.parent_path = parent_path
