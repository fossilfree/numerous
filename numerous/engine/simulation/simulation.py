import logging
from datetime import datetime
import time
import numpy as np
from numerous.engine.simulation.solvers.numerous_solver.numerous_solver import Numerous_solver
from numerous.engine.model import Model


class Simulation:
    """
    Class that wraps simulation solver. currently only solve_ivp.

    Attributes
    ----------
          time :  ndarray
               Not unique tag that will be used in reports or printed output.
          solver: :class:`BaseSolver`
               Instance of differential equations solver that will be used for the model.
          delta_t :  float
               timestep.
          callbacks :  list
               Not unique tag that will be used in reports or printed output.
          model :
               Not unique tag that will be used in reports or printed output.
    """

    def __init__(self, model: Model, t_start: float = 0, t_stop: float = 20000, num: int = 1000, num_inner: int = 1,
                 max_event_steps: int = 100,
                 start_datetime: datetime = datetime.now(), **kwargs):
        """
            Creating a namespace.

            Parameters
            ----------
            tag : string
                Name of a `VariableNamespace`

            Returns
            -------
            new_namespace : `VariableNamespace`
                Empty namespace with given name
        """
        if model.numba_model:
            model._reset()

        time_, delta_t = np.linspace(t_start, t_stop, num + 1, retstep=True)
        self.callbacks = []
        self.time = time_
        self.model = model

        logging.info("Generating Numba Model")
        generation_start = time.time()
        logging.info(f'Number of steps: {len(self.time)}')
        self.numba_model = model.generate_compiled_model(t_start, len(self.time))

        generation_finish = time.time()
        logging.info(f"Numba model generation finished, generation time: {generation_finish - generation_start}")

        event_function, event_directions = model.generate_event_condition_ast()
        event_function_full, _ = model.generate_event_condition_ast()
        action_function = model.generate_event_action_ast(model.events)
        if len(model.timestamp_events) == 0:
            model.generate_mock_timestamp_event()
        timestamp_action_function = model.generate_event_action_ast(model.timestamp_events)
        timestamps = np.array([np.array(event.timestamps) for event in model.timestamp_events])
        self.solver = Numerous_solver(time_, delta_t, model, self.numba_model,
                                      num_inner, max_event_steps, self.model.states_as_vector,
                                      numba_compiled_solver=model.use_llvm,
                                      events=(event_function, action_function), event_directions=event_directions,
                                      timestamp_events=(timestamp_action_function, timestamps),
                                      **kwargs)

        self.start_datetime = start_datetime
        self.info = model.info["Solver"]
        self.info["Number of Equation Calls"] = 0

        logging.info("Compiling Numba equations and initializing historian")
        compilation_start = time.time()
        self.numba_model.func(t_start, self.numba_model.get_states())
        self.numba_model.historian_update(t_start)
        compilation_finished = time.time()
        logging.info(
            f"Numba equations compiled, historian initizalized, compilation time: {compilation_finished - compilation_start}")

        self.compiled_model = self.numba_model

    def solve(self):
        self.reset()

        sol, self.result_status = self.solver.solve()

        self.info.update({"Solving status": self.result_status})
        self.complete()
        return sol

    def reset(self):
        self.__init_step()

        self.model.historian_df = None
        self.model.numba_model.historian_reinit()

        self.numba_model.historian_update(0)
        self.numba_model.map_external_data(0)

        if self.numba_model.is_external_data_update_needed(0):
            self.numba_model.is_external_data = self.model.external_mappings.load_new_external_data_batch(0)
        if self.numba_model.is_external_data:
            self.numba_model.update_external_data(self.model.external_mappings.external_mappings_numpy,
                                                  self.model.external_mappings.external_mappings_time,
                                                  self.model.external_mappings.t_max)

    def complete(self):

        list(map(lambda x: x.restore_variables_from_numba(self.solver.numba_model,
                                                          self.model.path_variables), self.model.callbacks))
        self.model.create_historian_df()

    def step_solve(self, t_start, step_size):
        try:
            t, results_status = self.solver.solver_step(t_start, step_size)

            return t, results_status
        except Exception as e:
            raise e

    def __init_step(self):
        pass
