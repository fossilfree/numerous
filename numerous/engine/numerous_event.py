class NumerousEvent:
    def __init__(self, key, condition, action, compiled,terminal,direction,compiled_functions=None):
        self.key = key
        self.condition = condition
        self.action = action
        self.compiled = compiled
        self.direction = direction
        self.terminal =terminal