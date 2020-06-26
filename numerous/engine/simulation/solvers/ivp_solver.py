import time

from scipy.integrate import solve_ivp
from tqdm import tqdm
import numpy as np

from numerous.engine.simulation.solvers.base_solver import BaseSolver


"""
Wraper for scipy ivp solver.
"""
class IVP_solver(BaseSolver):

    def __init__(self, time, delta_t, numba_model, num_inner, max_event_steps, **kwargs):
        super().__init__()
        self.time = time
        self.num_inner = num_inner
        self.delta_t = delta_t
        self.numba_model = numba_model
        self.diff_function = numba_model.func
        self.max_event_steps = max_event_steps
        self.options = kwargs

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
            solve_start = time.time()
            for t in tqdm(self.time[1:-1]):
                if self.solver_step(t):
                    break
            solve_finished = time.time()
            print("Solve time: ", solve_finished - solve_start)

        except Exception as e:
            print(e)
            raise e
        finally:
            return  self.sol,  self.result_status


    def solver_step(self,t):
        step_not_finished = True
        current_timestamp = t
        event_steps = 0


        stop_condition = False

        while step_not_finished:
            t_eval = np.linspace(current_timestamp, t + self.delta_t, self.num_inner + 1)

            self.sol = solve_ivp(self.diff_function, (current_timestamp, t + self.delta_t), y0=self.y0, t_eval=t_eval,
                            dense_output=True,
                            **self.options)
            step_not_finished = False
            event_step = self.sol.status == 1

            if self.sol.status == 0:
                current_timestamp = t + self.delta_t
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

                self.__end_step(self, sol.self.sol(current_timestamp), current_timestamp, event_id=event_id)
            else:
                if self.sol.success:
                    self.__end_step(self, self.sol.y[:, -1], current_timestamp)
                else:
                    self.result_status = self.sol.message
            if stop_condition:
                break
        if stop_condition:
            self.result_status = "Stopping condition reached"
            return True
        return False


    def set_state_vector(self, states_as_vector):
        self.y0 = states_as_vector

    def register_endstep(self, __end_step):
        self.__end_step =__end_step


