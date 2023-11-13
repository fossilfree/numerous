from datetime import datetime
import time
import numpy as np
from numerous.engine.simulation.solver_interface import generate_numerous_engine_solver_model
from numerous.engine.model import Model
from numerous.utils import logger as log
from numerous.solver.numerous_solver import NumerousSolver

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

    def __init__(self, model: Model, t_start: float = 0, t_stop: float = 20000, num: int = 1000,
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

        time_, delta_t = np.linspace(t_start, t_stop, num + 1, retstep=True)
        self.time = time_
        self.model = model

        if model.numba_model:
            self.reset(t_start)

        log.info(f'Number of steps: {len(self.time)}')
        if not model.numba_model:
            log.info("Generating Numba Model")
            generation_start = time.time()
            model._generate_compiled_model(t_start, len(self.time))

            generation_finish = time.time()
            log.info(f"Numba model generation finished, generation time: {generation_finish - generation_start}")

        numerous_engine_model, numerous_engine_event_handler = \
            generate_numerous_engine_solver_model(model)

        self.solver = NumerousSolver(model=numerous_engine_model,
                                     use_jit=model.use_llvm, event_handler=numerous_engine_event_handler,
                                     **kwargs)

        self.start_datetime = start_datetime
        self.info = model.info["Solver"]
        self.info["Number of Equation Calls"] = 0

        log.info("Compiling Numba equations and initializing historian")
        compilation_start = time.time()
        compilation_finished = time.time()
        log.info(
            f"Numba equations compiled, historian initizalized, compilation time: {compilation_finished - compilation_start}")

    def solve(self):

        self.solver.solve(self.time)
        self._complete()

    def reset(self, t_start: float):
        """
        Method which resets the simulation and model states to their initial values

        :param t_start:
        :return:
        """
        self.model._reset()
        self.__init_step()

        self.model.historian_df = None
        self.model.numba_model.historian_reinit()

        self.model.numba_model.map_external_data(t_start)
        self.model.numba_model.func(t_start, self.model.numba_model.get_states())

        if self.model.numba_model.is_external_data_update_needed(t_start):
            self.model.numba_model.is_external_data = self.model.external_mappings.load_new_external_data_batch(t_start)
        if self.model.numba_model.is_external_data:
            self.model.numba_model.update_external_data(self.model.external_mappings.external_mappings_numpy,
                                                        self.model.external_mappings.external_mappings_time,
                                                        self.model.external_mappings.t_max,
                                                        self.model.external_mappings.t_min)

    def _complete(self):

        list(map(lambda x: x.restore_variables_from_numba(self.model.numba_model,
                                                          self.model.path_variables), self.model.callbacks))
        self.model.create_historian_df()
        self._run_after()

    def _run_after(self):
        for function in self.model.run_after_solve:
            function()

    def step_solve(self, t_start: float, step_size: float):
        """
        making one simulation step
        Parameters
        ----------
        t_start : starting time
        step_size : simulation step size

        Returns
        -------

        """

        try:
            t, results_status = self.solver.solver_step(t_start, step_size)

            return t, results_status
        except Exception as e:
            raise e

    def __init_step(self):
        pass
