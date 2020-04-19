import time
from datetime import datetime
import time
import numpy as np
from numba import njit, prange
from scipy.integrate import solve_ivp
from tqdm import tqdm


class Simulation:
    """
    Class that wraps simulation solver. currently only solve_ivp.

    Attributes
    ----------
          time :  ndarray
               Not unique tag that will be used in reports or printed output.
          delta_t :  float
               timestep.
          callbacks :  list
               Not unique tag that will be used in reports or printed output.
          model :
               Not unique tag that will be used in reports or printed output.
    """

    def __init__(self, model, t_start=0, t_stop=20000, num=1000, num_inner=1, max_event_steps=100,
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

        self.time, self.delta_t = np.linspace(t_start, t_stop, num + 1, retstep=True)
        self.callbacks = []
        self.async_callback = []
        self.model = model
        self.eq_count = np.unique(self.model.compiled_eq_idxs).shape[0]
        self.sum_mapping = self.model.sum_idx.size != 0
        self.start_datetime = start_datetime
        self.num_inner = num_inner
        self.options = kwargs
        self.max_event_steps = max_event_steps
        self.info = model.info["Solver"]
        self.info["Number of Equation Calls"] = 0
        self.y0 = self.model.states_as_vector
        if self.y0.size == 0:
            self.__func = self.stateless__func
        self.events = [model.events[event_name].event_function._event_wrapper() for event_name in model.events]
        self.callbacks = [x.callbacks for x in sorted(model.callbacks,
                                                      key=lambda callback: callback.priority,
                                                      reverse=True)]
        self.t_scope = self.model._get_initial_scope_copy()

    def __end_step(self, y, t, event_id=None, **kwargs):
        self.model.update_model_from_scope(self.t_scope)
        self.model.sychronize_scope()
        for callback in self.callbacks:
            callback(t, self.model.path_variables, **kwargs)
        if event_id is not None:
            list(self.model.events.items())[event_id][1]._callbacks_call(t, self.model.path_variables)
        # self.model.update_flat_scope(self.t_scope)

        #self.model.sychronize_scope()

        self.y0 = self.model.states_as_vector

    def __init_step(self):
        [x.initialize(simulation=self) for x in self.model.callbacks]


