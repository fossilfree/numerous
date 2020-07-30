import sys
import time
from copy import deepcopy
import numpy as np
from numba import njit
from tqdm import tqdm

from numerous.engine.simulation.solvers.base_solver import BaseSolver
from simulation.solvers.numerous_solver.solver_methods import LevenbergMarquardt


class Numerous_solver(BaseSolver):

    def __init__(self, time, delta_t, numba_model, num_inner, max_event_steps, **kwargs):
        super().__init__()
        self.time = time
        self.num_inner = num_inner
        self.delta_t = delta_t
        self.numba_model = numba_model
        self.diff_function = numba_model.func
        self.max_event_steps = max_event_steps
        self.options = kwargs
        dt = 0.1
        odesolver_options = {'dt': dt, 'longer': 1, 'shorter': 1, 'min_dt': dt, 'strict_eval': False, 'max_step': dt,
                             'first_step': dt, 'atol': 1e-6, 'rtol': 1e-3, 'order': 0, 'outer_itermax': 20}
        odesolver_options_bde = deepcopy(odesolver_options)
        odesolver_options_bde['order'] = 5
        self.method_options = odesolver_options_bde
        self.method = LevenbergMarquardt


    def generate_solver(self):
        eps = np.finfo(1.0).eps

        # Set options
        longer = self.method_options.get('longer', 1.2)
        shorter = self.method_options.get('shorter', 0.8)

        max_step = self.method_options.get('max_step', 3)
        min_step = self.method_options.get('min_step', 10 * eps)

        initial_step = max_step  # np.min([100000000*min_step, max_step])

        history_buffer_length = self.method_options.get('history_buffer_length', 1000)

        order = self.method_options.get('order', 0)

        strict_eval = self.method_options.get('strict_eval', True)
        outer_itermax = self.method_options.get('outer_itermax', 20)

        _method = self.method(**self.method_options)
        step_integrate_ = _method.step_func


        # @njit
        def _solve(numba_model, _solve_state, t_end=1000.0, t0=0.0, t_eval=np.linspace(0.0, 1000.0, 100), tol=0.001):
            # Init t to t0
            t = t0
            dt = initial_step / longer

            order_ = -1
            len_y = numba_model.get_states().shape[0]
            n = order + 2
            roller = (n, np.zeros(n), np.zeros((n, len_y)))
            roller_ix = -1

            def rollforward(t_, y_, rb, ix, o):
                o = min(o + 1, order)
                ix += 1
                if ix >= rb[0]:
                    ix = 0
                rb[1][ix] = t_
                rb[2][ix, :] = y_
                return ix, o

            def rollback(rb, ix):
                return ix, rb[1][ix], rb[2][ix, :]

            def t_rollback(rb, ix):
                if ix < 0:
                    ix = rb[0] - 1 + ix
                return rb[1][ix]

            # Define event derivatives, values and event guess times
            # edt = np.ones(max(1, n_events)) * tol + 2.0
            # et = np.zeros(n_events) if n_events > 0 else np.ones(1)
            te_array = np.zeros(2)
            # 0 index is used to keep next time step defined by solver
            te_array[0] = t
            # 1 index is used to keep next time to eval/save the solution
            ix_eval = 1
            te_array[1] = t_eval[ix_eval] + dt if strict_eval else np.inf
            t_next_eval = t_eval[ix_eval]

            outer_iter = outer_itermax

            y = numba_model.get_states()

            terminate = False
            step_converged = True
            dt_last = -1
            j_i = 0
            progress_c = t_eval.shape[0]
            while not terminate:
                # updated events time estimates
                # # time acceleration
                if not step_converged:
                    dt *= shorter
                    if min_step > dt:
                        raise ValueError('dt shortened below min_dt of: ', min_step)
                    decrease = True
                    te_array[0] = t_rollback(roller, roller_ix) + dt
                else:
                    dt = min(max_step, dt * longer)
                    decrease = False

                    te_array[0] = t + dt

                # Determine new test - it should be the smallest value requested by events, eval, step
                t_new_test = np.min(te_array)

                # Determine if rollback is needed
                # check if t_new_test is forward
                # no need to roll back if time-step is decreased, as it's not a failed step
                if (t_new_test < t) or (not step_converged):
                    # Need to roll back!
                    roller_ix, t_start, y = rollback(roller, roller_ix)
                    if t_new_test < t_start:
                        # t_new_test = t_rollback
                        # TODO: make more specific error raising here!
                        raise ValueError('Cannot go back longer than rollback point!')
                else:
                    # Since we didnt roll back we can update t_start and rollback
                    # Check if we should update history at t eval
                    if t_next_eval <= t:
                        j_i+=1
                        p_size = 100
                        x = int(p_size * j_i / progress_c)
                        print(x)
                        numba_model.historian_update(t)
                        if strict_eval:
                            te_array[1] = t_next_eval = t_eval[ix_eval + 1] if ix_eval + 1 < len(t_eval) else t_eval[-1]
                        else:
                            t_next_eval = t_eval[ix_eval + 1] if ix_eval + 1 < len(t_eval) else t_eval[-1]
                        ix_eval += 1


                    roller_ix, order_ = rollforward(t, y, roller, roller_ix, order_)
                    t_start = t
                    t_new_test = np.min(te_array)
                    if t >= t_end:
                        break
                    # t_rollback = t
                    # y_rollback = y_previous
                dt_ = t_new_test - t_start
                if dt_ != dt_last:
                    # print('print new dt: ', dt_)
                    dt_last = dt_
                    order_ = 0
                    # solve from start to new test by calling the step function
                t, y, step_converged, step_info, _solve_state = step_integrate_(numba_model,
                                                                                    t_start,
                                                                                    dt_, y, _solve_state)

            info = {'step_info': step_info}
            # Return the part of the history actually filled in
            # print(history[1:,0])
            return info

        self._method = _method

        # Return the solver function
        return _solve

    def set_state_vector(self, states_as_vector):
        self.y0 = states_as_vector

    def solve(self):
        """
        solve the model.

        Returns
        -------
        Solution : 'OdeSoulution'
                returns the most recent OdeSolution from scipy

        """
        self.result_status = "Success"
        self.sol = None

        try:

            t_start = self.time[0]
            t_end = self.time[-1]
            # Generate the solver - if it already is generated its just returned.
            _solve = self.generate_solver()

            # Call the solver
            from copy import deepcopy
            y0 = deepcopy(self.y0)
            state = deepcopy(self.y0)

            # figure out solve_state init
            solve_state = self._method.get_solver_state(len(y0))
            info = _solve(self.numba_model,
                          solve_state, t0=t_start, t_end=t_end, t_eval=self.time)
            print("finished")

        except Exception as e:
            print(e)
            raise e
        finally:
            return self.sol, self.result_status


    def register_endstep(self, __end_step):
        self.__end_step =__end_step
