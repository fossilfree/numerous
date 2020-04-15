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

    def get_diff(self):
        return self.__func

    def get_diff_(self):
        # Method that returns the differentiation function

        @njit
        def compute_eq(self, array_2d):
            for eq_idx in range(self.eq_count):
                self.model.compiled_eq[eq_idx](array_2d[eq_idx])

        @njit
        def compute(self):
            if self.sum_mapping:
                sum_mappings(self.model.sum_idx, self.model.sum_mapped_idx, self.t_scope.flat_var,
                             self.model.sum_mapped)
            mapping_ = True
            b1 = np.copy(self.t_scope.flat_var)
            while mapping_:
                mapping_from(self.model.compiled_eq_idxs, self.model.index_helper, self.model.scope_variables_2d,
                             self.model.length, self.t_scope.flat_var, self.model.flat_scope_idx_from,
                             self.model.flat_scope_idx_from_idx_1, self.model.flat_scope_idx_from_idx_2)

                self.compute_eq(self.model.scope_variables_2d)

                mapping_to(self.model.compiled_eq_idxs, self.t_scope.flat_var, self.model.flat_scope_idx,
                           self.model.scope_variables_2d,
                           self.model.index_helper, self.model.length,
                           self.model.flat_scope_idx_idx_1, self.model.flat_scope_idx_idx_2)

                if self.sum_mapping:
                    sum_mappings(self.model.sum_idx, self.model.sum_mapped_idx, self.t_scope.flat_var,
                                 self.model.sum_mapped)

                mapping_ = not np.allclose(b1, self.t_scope.flat_var)
                b1 = np.copy(self.t_scope.flat_var)

        @njit
        def __func(self, _t, y, state):

            self.info["Number of Equation Calls"] += 1

            self.t_scope.update_states(y)
            self.model.global_vars[0] = _t

            self.compute()

            return self.t_scope.get_derivatives()

        return __func

    def get_initial_states(self):
        # Method that returns the initial state vector
        return self.y0

    def get_internal_state(self):
        return self.state

    def post_process_history(self, t, history, final):
        # Method that is used to post processing the history before it is handed to the obj mode history
        pass

@njit()
def mapping_to(compiled_eq_idxs, flat_var, flat_scope_idx, scope_variables_2d, index_helper, length, id1, id2):
    for i in prange(compiled_eq_idxs.shape[0]):
        eq_idx = compiled_eq_idxs[i]
        flat_var[flat_scope_idx[id1[i]:id2[i]]] = \
            scope_variables_2d[eq_idx][index_helper[i]][:length[i]]

@njit()
def mapping_from(compiled_eq_idxs, index_helper, scope_variables_2d, length, flat_var, flat_scope_idx_from, id1,
                 id2):
    for i in prange(compiled_eq_idxs.shape[0]):
        eq_idx = compiled_eq_idxs[i]
        scope_variables_2d[eq_idx][index_helper[i]][:length[i]] \
            = flat_var[flat_scope_idx_from[id1[i]:id2[i]]]

@njit()
def sum_mappings(sum_idx, sum_mapped_idx, flat_var, sum_mapped):
    for i in prange(sum_idx.shape[0]):
        idx = sum_idx[i]
        slice_ = sum_mapped_idx[i]
        flat_var[idx] = np.sum(flat_var[sum_mapped[slice_[0]:slice_[1]]])

if __name__ == "__main__":
    from numerous.engine import model
    from matplotlib import pyplot as plt
    from numerous_solver.solver_methods import LevenbergMarquardt, BackwardEuler
    from numerous_solver.simulation import Simulation
    from examples.dampened_oscillator.dampened_oscillator import OscillatorSystem
    import numpy as np

    dt=0.1
    t_end = 100
    t_eval = np.linspace(0, t_end, 10 + 1)

    odesolver_options={'dt': dt, 'longer': 1, 'shorter': 1, 'max_dt': dt, 'min_dt': t_end/1e6, 'abs_tol':1e-2, 'rel_tol': 1e-2, 'strict_eval': True, 'max_step': 1000}

    mw = ModelWrap(model.Model(OscillatorSystem('system')))

    s = Simulation(method=BackwardEuler, model=mw,
                               method_options=odesolver_options)

    # Solve and plot
    sol, tot_time, rate = s.solve(t_end=t_end, t_eval=t_eval)
    s.model.historian.df.plot()
    plt.show()
    plt.interactive(False)