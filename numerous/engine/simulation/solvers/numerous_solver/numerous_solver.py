import time
from copy import deepcopy
import numpy as np
from numba import njit
from tqdm import tqdm

from numerous.engine.simulation.solvers.base_solver import BaseSolver
from numerous.engine.simulation.solvers.numerous_solver.solver_methods import BDF5


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
        self.method_options=odesolver_options_bde
        self.method = BDF5

        self._solve =self.solver.generate_solver()

    def generate_solver(self):
            eps = np.finfo(1.0).eps

            # Set options
            longer = self.method_options.get('longer', 1.2)
            shorter = self.method_options.get('shorter', 0.8)

            max_step = self.method_options.get('max_step', 3)
            min_step = self.method_options.get('min_step', 10*eps)

            initial_step = max_step


            order = self.method_options.get('order', 0)

            strict_eval = self.method_options.get('strict_eval', True)
            outer_itermax = self.method_options.get('outer_itermax', 20)
            _method = self.method(**self.method_options)
            step_integrate_ = _method.step_func
            diff_ = self.diff_function


            @njit
            def _solve(_solve_state, _state, y0, t_end=1000.0, t0=0.0, t_eval=np.linspace(0.0, 1000.0, 100), tol=0.001):
                # Init t to t0
                t = t0
                dt = initial_step / longer
                order_ = -1
                len_y = len(y0)
                # Define history
                n = order+2
                roller = (n, np.zeros(n), np.zeros((n, len(y0))))
                roller_ix = -1

                def rollforward(t_, y_, rb, ix, o):

                    o = min(o+1, order)
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

                def get_order_y(rb, ix, order):
                    ix_ = ix-1

                    if ix_<0:

                        return rb[2][-order - 1:, :]
                    elif order > ix:
                        yo = np.zeros((order, len_y))
                        # add first lines
                        if ix_>=0:
                        #assign latest history lines
                            yo[-ix_-1:, :] = rb[2][:ix_+1, :]
                        # add last lines
                        yo[:-ix_-1, :] = rb[2][-order + ix_+1:, :]
                        return yo
                    else:
                        yo = np.zeros((ix_+1, len_y))
                        yo = rb[2][:ix_+1]

                        return yo
                # 1 index is used to keep next time to eval/save the solution
                ix_eval = 0
                t_next_eval = t_eval[ix_eval]

                outer_iter = outer_itermax

                # Define the t to rollback to if event value requires it or solver not converging
                # t_rollback =
                t_previous_test = t0
                # Define state vectors
                y = np.copy(y0)

                terminate = False
                step_converged = True
                dt_last = -1
                while not terminate:
                    # time acceleration
                    if not step_converged:
                        dt *= shorter
                        if min_step > dt:
                            raise ValueError('dt shortened below min_dt of: ', min_step)
                        decrease = True
                    else:
                        dt = min(max_step, dt * longer)
                        decrease = False
                    if not step_converged:
                        roller_ix, t_start, y = rollback(roller, roller_ix)
                    else:
                        # Since we didnt roll back we can update t_start and rollback
                        # Check if we should update history at t eval

                        if t_next_eval <= t:
                             # update_history(output_func_, _state, history, ix_history, t, y)

                            # Set next t eval
                            if strict_eval:
                                t_next_eval = t_eval[ix_eval+1] if ix_eval+1 < len(t_eval) else t_eval[-1]
                            else:
                                t_next_eval = t_eval[ix_eval+1] if ix_eval+1 < len(t_eval) else t_eval[-1]
                            ix_eval += 1

                        # Check if there is some callbacks to call
                        self.numba_model.historian_update(t)

                        roller_ix, order_ = rollforward(t, y, roller, roller_ix, order_)
                        t_start = t

                    dt_ = t_new_test - t_start
                    if dt_ != dt_last:
                        #print('print new dt: ', dt_)
                        dt_last = dt_
                        order_ = 0
                    # solve from start to new test by calling the step function
                    if order > 1:
                        t, y, step_info, outer_iter = step_integrate_(diff_, diff_single_, _state, t_start,
                                                                  dt_, y, get_order_y(roller, roller_ix, order_), order_)
                    else:
                        t, y, step_converged, step_info, _solve_state = step_integrate_(diff_, diff_single_, _state, t_start,
                                                                      dt_, y, _solve_state)
                    # y_previous = y
                    # t_previous_test = t

                # Build the info to return
                info = {'step_info': step_info}
                # Return the part of the history actually filled in
                # print(history[1:,0])
                return info

            self._method = _method

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
            print("Compiling Numba equations")
            compilation_start = time.time()
            self.solver_step(self.time[0])
            compilation_finished = time.time()
            print("Compilation time: ", compilation_finished - compilation_start)
            for t in tqdm(self.time[1:-1]):
                if self.solver_step(t):
                    break
        except Exception as e:
            print(e)
            raise e
        finally:
            return self.sol, self.result_status