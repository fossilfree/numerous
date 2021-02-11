from datetime import datetime
import time
import numpy as np
from numerous.engine.simulation.solvers.base_solver import SolverType
from numerous.engine.simulation.solvers.ivp_solver.ivp_solver import IVP_solver
from .solvers.numerous_solver.numerous_solver import Numerous_solver


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

    def __init__(self, model, solver_type=SolverType.SOLVER_IVP, t_start=0, t_stop=20000, num=1000, num_inner=1,
                 max_event_steps=100,

                 start_datetime=datetime.now(), **kwargs):
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
        self.callbacks = []
        self.time = time_
        self.async_callback = []
        self.model = model

        def __end_step(solver, y, t):
            """

            """

            solver.y0 = y

            solver.numba_model.historian_update(t)
            # solver.numba_model.run_callbacks_with_updates(t)
            solver.numba_model.map_external_data(t)

            if solver.numba_model.is_store_required():
                self.model.store_history(numba_model.historian_data)
                solver.numba_model.historian_reinit()
            #
            if solver.numba_model.is_external_data_update_needed(t):
                solver.numba_model.is_external_data = self.model.external_mappings.load_new_external_data_batch(t)
                if solver.numba_model.is_external_data:
                    solver.numba_model.update_external_data(self.model.external_mappings.external_mappings_numpy,
                                                            self.model.external_mappings.external_mappings_time)

        self.end_step = __end_step
        print("Generating Numba Model")
        generation_start = time.time()
        numba_model = model.generate_compiled_model(t_start, len(self.time))

        generation_finish = time.time()
        print("Generation time: ", generation_finish - generation_start)

        if solver_type.value == SolverType.SOLVER_IVP.value:
            self.solver = IVP_solver(time_, delta_t, model, numba_model,
                                     num_inner, max_event_steps, self.model.states_as_vector, **kwargs)

        if solver_type.value == SolverType.NUMEROUS.value:
            self.solver = Numerous_solver(time_, delta_t,model,  numba_model,
                                          num_inner, max_event_steps, self.model.states_as_vector,
                                          numba_compiled_solver=model.use_llvm, **kwargs)

        self.solver.register_endstep(__end_step)

        self.start_datetime = start_datetime
        self.info = model.info["Solver"]
        self.info["Number of Equation Calls"] = 0

        print("Compiling Numba equations and initializing historian")
        compilation_start = time.time()
        numba_model.func(t_start, numba_model.get_states())
        numba_model.historian_update(t_start)
        compilation_finished = time.time()
        print("Compilation time: ", compilation_finished - compilation_start)

        self.compiled_model = numba_model
        # self.solver.events = [model.events[event_name].event_function._event_wrapper() for event_name in model.events]
        # self.callbacks = [x.callbacks for x in sorted(model.callbacks,
        #                                               key=lambda callback: callback.priority,
        #                                               reverse=True)]

    def solve(self):
        self.reset()

        result_status = "not finished"
        try:
            sol, self.result_status = self.solver.solve()

        except Exception as e:
            raise e

        finally:
            self.info.update({"Solving status": self.result_status})
            self.complete()
        return sol

    def reset(self):
        self.__init_step()
        self.model.numba_model.historian_reinit()

        self.end_step(self.solver, self.model.numba_model.get_states(), 0)

    def step(self, dt):
        try:
            stop = self.solver.solver_step(dt)
        except Exception as e:
            raise e

        return stop

    def complete(self):

        list(map(lambda x: x.restore_variables_from_numba(self.solver.numba_model,
                                                          self.model.path_variables), self.model.callbacks))
        self.model.create_historian_df()

    def step_solve(self, t, step_size):
        try:
            t, results_status = self.solver.solver_step(t, step_size)

            return t, results_status
        except Exception as e:
            raise e

    def __init_step(self):
        pass
        # [x.initialize(simulation=self) for x in self.model.callbacks]
