class NumerousEvent:
    def __init__(self, key, condition, action, compiled, terminal, direction, condition_full=None):
        self.key = key
        self.condition = condition
        self.condition_full = condition_full
        self.action = action
        self.compiled = compiled
        self.direction = direction
        self.terminal = terminal


class TimestampEvent:
    def __init__(self, key, action, timestamps):
        self.key = key
        self.action = action
        self.timestamps = timestamps
