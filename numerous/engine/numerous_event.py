class NumerousEvent:

    def __init__(self, key, condition, action, compiled, terminal, direction, compiled_functions=None):
        self.key = key
        self.condition = condition
        self.action = action
        self.compiled = compiled
        self.compiled_functions = compiled_functions
        self.direction = direction
        self.terminal = terminal


class TimestampEvent:
    def __init__(self, key, action, timestamps=None, periodicity=None, compiled_functions=None):
        self.key = key
        self.action = action
        self.compiled_functions = compiled_functions

        assert not (periodicity and timestamps), "you cannot specify both timestamps and periodicity for a " \
                                                 "time-stamp event"

        self.timestamps = timestamps
        self.periodicity = periodicity
