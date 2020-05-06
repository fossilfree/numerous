from datetime import datetime
import time
import numpy as np
from engine.simulation.solvers.base_solver import SolverType
from engine.simulation.solvers.ivp_solver import IVP_solver


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
            self.solver = IVP_solver(time, delta_t, model.get_diff_(), num_inner,max_event_steps,**kwargs)
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

    def compute_eq(self, array_2d):
        for eq_idx in range(self.eq_count):
            self.model.compiled_eq[eq_idx](array_2d[eq_idx,:self.model.num_uses_per_eq[eq_idx]])

    def __func(self, _t, y):
        self.info["Number of Equation Calls"] += 1
        self.t_scope.update_states(y)
        self.model.global_vars[0] = _t

        self.compute()

        return self.t_scope.get_derivatives()

    def compute(self):
        if self.sum_mapping:
            sum_mappings(self.model.sum_idx, self.model.sum_mapped_idx, self.t_scope.scope_vars_3d,
                         self.model.sum_mapped)
        mapping_ = True
        prev_scope_vars_3d = self.t_scope.scope_vars_3d.copy()
        while mapping_:

            self.t_scope.scope_vars_3d[self.model.differing_idxs_pos_3d] = self.t_scope.scope_vars_3d[self.model.differing_idxs_from_3d]
            self.compute_eq(self.t_scope.scope_vars_3d)
           
            if self.sum_mapping:
                sum_mappings(self.model.sum_idx, self.model.sum_mapped_idx, self.t_scope.scope_vars_3d,
                             self.model.sum_mapped)

            mapping_ = not np.allclose(prev_scope_vars_3d, self.t_scope.scope_vars_3d)
            np.copyto(prev_scope_vars_3d, self.t_scope.scope_vars_3d)

    def stateless__func(self, _t, _):
        self.info["Number of Equation Calls"] += 1
        self.compute()
        return np.array([])

def sum_mappings(sum_idx, sum_mapped_idx, flat_var, sum_mapped):
    raise ValueError
    for i in prange(sum_idx.shape[0]):
        idx = sum_idx[i]
        slice_ = sum_mapped_idx[i]
        flat_var[idx] = np.sum(flat_var[sum_mapped[slice_[0]:slice_[1]]])

        # self.model.update_model_from_scope(self.t_scope)
        # self.model.sychronize_scope()
        # for callback in self.callbacks:
        #     callback(t, self.model.path_variables, **kwargs)
        # if event_id is not None:
        #     list(self.model.events.items())[event_id][1]._callbacks_call(t, self.model.path_variables)
        self.model.synchornize_scope()
        for callback in self.callbacks:
            callback(t, self.model.path_variables, **kwargs)
        # if event_id is not None:
        #     list(self.model.events.items())[event_id][1]._callbacks_call(t, self.model.path_variables)

