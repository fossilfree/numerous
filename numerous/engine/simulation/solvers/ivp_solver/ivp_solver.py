import time

from scipy.integrate import solve_ivp
from tqdm import tqdm
import numpy as np

from numerous.engine.simulation.solvers.base_solver import BaseSolver


"""
Wrapper for scipy ivp solver.
"""
class IVP_solver(BaseSolver):

    def __init__(self, time, delta_t, model, numba_model, num_inner, max_event_steps, y0, **kwargs):
        super().__init__()
        self.model = model
        self.time = time
        self.y0=y0
        self.num_inner = num_inner
        self.delta_t = delta_t
        self.numba_model = numba_model
        self.diff_function = numba_model.func
        self.max_event_steps = max_event_steps
        self.options = kwargs
        self.t = 0

    def solve(self):
        """
        solve the model.
        Returns
        -------
        Solution : 'OdeSolution'
                returns the most recent OdeSolution from scipy
        """
        self.result_status = "Success"
        self.sol = None

        try:
            for t in tqdm(self.time[0:-1]):
                if self.solver_step(t):
                    break
        except Exception as e:
            print(e)
            raise e

        return self.sol,  self.result_status

    def solver_step(self, t, delta_t=None):
        step_not_finished = True
        current_timestamp = t
        event_steps = 0

        step_solver_mode = False
        if delta_t is None:
            delta_t = self.delta_t
        else:
            step_solver_mode = True

        stop_condition = False
        t_end = current_timestamp + delta_t

        while step_not_finished:
            t_eval = np.linspace(current_timestamp, t_end, self.num_inner + 1)

            self.sol = solve_ivp(self.diff_function, (current_timestamp, t + delta_t), y0=self.y0, t_eval=t_eval,
                            dense_output=False,
                            **self.options)
            step_not_finished = False
            event_step = self.sol.status == 1

            if self.sol.status == 0:
                current_timestamp = t_end
                if step_solver_mode: # added this
                    self.model.numba_model.historian_update(current_timestamp)
                    self.y0 = self.sol.y[:, -1]
                    return current_timestamp, self.sol.t[-1]
            if event_step:
                event_id = np.nonzero([x.size > 0 for x in self.sol.t_events])[0][0]
                # solution stuck
                stop_condition = False
                if (abs(self.sol.t_events[event_id][0] - current_timestamp) < 1e-6):
                    event_steps += 1
                else:
                    event_steps = 0

                if event_steps > self.max_event_steps:
                    stop_condition = True
                current_timestamp = self.sol.t_events[event_id][0]
                step_not_finished = True
                self.__end_step(self, self.sol(current_timestamp), current_timestamp, event_id=event_id)
            else:
                if self.sol.success:
                    self.__end_step(self, self.sol.y[:, -1], current_timestamp)
                else:
                    self.result_status = self.sol.message
            if stop_condition:
                break

        self.t = t_end
        if stop_condition:
            self.result_status = "Stopping condition reached"
            return True
        return False


    def set_state_vector(self, states_as_vector):
        self.y0 = states_as_vector

    def register_endstep(self, __end_step):
        self.__end_step =__end_step