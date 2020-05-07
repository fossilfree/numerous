from datetime import datetime
import numpy as np
from numerous.engine.simulation.solvers.base_solver import SolverType
from numerous.engine.simulation.solvers.ivp_solver import IVP_solver


class Simulation:
    """
    Class that wraps simulation solver. currently only solve_ivp.

    Attributes
    ----------
          time :  ndarray
               Not unique tag that will be used in reports or printed output.
          solver: :class:`BaseSolver`
               Instance of differantial eqautions solver that will be used for the model.
          delta_t :  float
               timestep.
          callbacks :  list
               Not unique tag that will be used in reports or printed output.
          model :
               Not unique tag that will be used in reports or printed output.
    """

    def __init__(self, model, solver_type=SolverType.SOLVER_IVP, t_start=0, t_stop=20000, num=1000, num_inner=1, max_event_steps=100,

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

        time,delta_t = np.linspace(t_start, t_stop, num + 1, retstep=True)
        self.callbacks = []
        self.time = time
        self.async_callback = []
        if solver_type == SolverType.SOLVER_IVP:
            self.solver = IVP_solver(time, delta_t, model.numba_model, num_inner,max_event_steps,**kwargs)
        self.model = model
        self.start_datetime = start_datetime
        self.info = model.info["Solver"]
        self.info["Number of Equation Calls"] = 0
        self.solver.set_state_vector(self.model.states_as_vector)
        self.solver.events = [model.events[event_name].event_function._event_wrapper() for event_name in model.events]
        self.callbacks = [x.callbacks for x in sorted(model.callbacks,
                                                      key=lambda callback: callback.priority,
                                                      reverse=True)]

        self.solver.register_endstep(self.__end_step)



    def solve(self):
        self.__init_step()
        try:
            sol, result_status = self.solver.solve()
        except Exception as e:
            raise e
        finally:
            self.info.update({"Solving status": result_status})
            list(map(lambda x: x.finalize(), self.model.callbacks))
        return sol

    def __init_step(self):
        [x.initialize(simulation=self) for x in self.model.callbacks]


    def __end_step(self, solver, y, t, event_id=None, **kwargs):
        solver.y0 = y

        # self.model.update_model_from_scope(self.t_scope)
        # self.model.sychronize_scope()
        # for callback in self.callbacks:
        #     callback(t, self.model.path_variables, **kwargs)
        # if event_id is not None:
        #     list(self.model.events.items())[event_id][1]._callbacks_call(t, self.model.path_variables)
        solver.numba_model.update_path_variables()
        # solver.numba_model.run_registered_callbacks()
        # self.model.synchornize_variables()
        # for callback in self.callbacks:
        #     callback(t, solver.numba_model.path_variables, **kwargs)
        # if event_id is not None:
        #     list(self.model.events.items())[event_id][1]._callbacks_call(t, self.model.path_variables)