from numerous_solver.interfaces import Model as NSM
from numerous_solver.compile_decorators import basic_njit as njit
from datetime import datetime


class ModelWrap(NSM):
    def __init__(self, model, start_datetime=datetime.now(), **kwargs):
        self.model = model
        self.eq_count = np.unique(self.model.compiled_eq_idxs).shape[0]
        self.sum_mapping = self.model.sum_idx.size != 0
        self.start_datetime = start_datetime

        self.y0 = self.model.states_as_vector

        if self.y0.size == 0:
            self.__func = self.stateless__func
        else:
            @njit
            def stateless__func(self, _t, _):
                self.info["Number of Equation Calls"] += 1
                self.compute()
                return np.array([])
            self.__func = stateless__func

        self.events = [model.events[event_name].event_function._event_wrapper() for event_name in model.events]
        self.callbacks = [x.callbacks for x in sorted(model.callbacks,
                                                      key=lambda callback: callback.priority,
                                                      reverse=True)]
        self.t_scope = self.model._get_initial_scope_copy()

    def get_compiled_callbacks(self):
        # Method used by Simulation to get the callbacks from the model
        # Should return a function representing the callbacks, the period between calling each callback and the number of callbacks
        @njit
        def dummy_callback(t, y, state, indices):
            return False

        return dummy_callback, [1], 0

    def get_compiled_events(self):
        # Method used by Simulation to get the events from the model
        # Should return a function representing the events, a function representing the callbacks and the number of events
        @njit
        def dummy_event(t, y, state, edt, et):
            pass

        @njit
        def dummy_callback(t, y, state):
            return False

        return dummy_event, dummy_callback, 0

    def get_output_func(self):
        # Method that returns the function that picks the output variables to be saved and place them in a array - this array is passed to the history callback
        @njit
        def output(t, y, state):
            return y

        return output


    def get_initial_states(self):
        # Method that returns the initial state vector
        return self.y0

    def get_internal_state(self):
        return self.state

    def post_process_history(self, t, history, final):
        # Method that is used to post processing the history before it is handed to the obj mode history
        pass
