class _SimulationCallback:

    def __init__(self, name,callback_function=None):
        self.name = name
        if callback_function:
            self.callback_functions = [callback_function]
        else:
            self.callback_functions = []
        self.finalize_functions = []
        self.initialize_functions = []



    def add_callback_function(self, callback_function):
        self.callback_functions.append(callback_function)

    def add_finalize_function(self, finalize_function):
        self.finalize_functions.append(finalize_function)

    def add_initialize_function(self, initialize_function):
        self.initialize_functions.append(initialize_function)

    def callbacks(self, t, variables):
        for callback in self.callback_functions:
            callback(t, variables)

    # Pass the simulation object for initialization only
    def initialize(self, simulation=object):
        for initialize in self.initialize_functions:
            initialize(simulation)

    def finalize(self):
        for finalize in self.finalize_functions:
            finalize()



class _EventFunction:
    """
    Wrapper around callable event.
    """
    def __init__(self, name, model, event_function=None):
        self.name = name
        self.model = model
        ##Only terminal events are supported
        self.event_function = event_function

    def _event_wrapper(self):
        def event(t, y):
            for variable in self.model.variables:
                if self.model.scope_variables[variable].state_ix is not None:
                    self.model.scope_variables[variable].value = y[self.model.scope_variables[variable].state_ix]
            return self.event_function(t, self.model.path_scope_variables)
        event.terminal = True
        event.direction = self.event_function.direction
        return event

class _Event:
    def __init__(self, name, model, event_function=None, callbacks=None):
        self.name = name
        self.model = model
        self.event_function = self.add_event(name, event_function)
        self.callbacks = callbacks

    def add_event(self, name, event_function):
        return _EventFunction(name, self.model, event_function)

    def add_callbacks(self, callback):
        self.callbacks.append(callback)

    def _callbacks_call(self, t, variables):
        for callback in self.callbacks:
            callback(t, variables)
