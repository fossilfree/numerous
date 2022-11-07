import numpy as np
import numba as nb
from numba.core.registry import CPUDispatcher
import numpy.typing as npt
from numba.experimental import jitclass
from numerous.engine.model.compiled_model import CompiledModel
from numerous.engine.model import Model
from numerous.engine.simulation.solvers.numerous_solver.interface import SolverInterface, \
    ModelInterface
from numerous.engine.simulation.solvers.numerous_solver.common import jithelper
from numerous.engine.simulation.solvers.numerous_solver.numerous_solver import SolveEvent


class NumerousEngineModelInterface(ModelInterface):
    def __init__(self, nm: CompiledModel, event_functions, event_directions, event_actions, time_events,
                 time_event_actions):
        self.nm = nm
        self.event_functions = event_functions
        self.event_actions = event_actions
        self.event_directions = event_directions
        self.time_events = time_events
        self.time_event_actions = time_event_actions

    def get_deriv(self, t) -> np.array:
        y = self.get_states()
        return self.nm.func(t, y)

    def get_states(self) -> np.array:
        return self.nm.get_states()

    def set_states(self, states: npt.ArrayLike) -> None:
        return self.nm.set_states(states)

    def _get_variables(self) -> np.array:
        return self.nm.read_variables()

    def _write_variables(self, value: float, idx: int) -> None:
        self.nm.write_variables(value, idx)

    def _is_store_required(self) -> bool:
        return self.nm.is_store_required()

    def _is_external_data_update_needed(self, t) -> bool:
        return self.nm.is_external_data_update_needed(t)

    def _map_external_data(self, t) -> None:
        return self.nm.map_external_data(t)

    def historian_update(self, t: np.float64) -> SolveEvent:
        self.nm.historian_update(t)
        if self._is_store_required():
            return SolveEvent.Historian
        else:
            return SolveEvent.NoneEvent

    def pre_step(self, t) -> None:
        self._map_external_data(t)

    def post_step(self, t) -> SolveEvent:
        self._map_external_data(t)

        if self._is_store_required() and not self._is_external_data_update_needed(t):
            return SolveEvent.Historian
        elif not self._is_store_required() and self._is_external_data_update_needed(t):
            return SolveEvent.ExternalDataUpdate
        elif self._is_store_required() and self._is_external_data_update_needed(t):
            return SolveEvent.HistorianAndExternalUpdate
        else:
            return SolveEvent.NoneEvent

    def post_event(self, t: np.float64) -> SolveEvent:
        return self.historian_update(t)

    def get_event_results(self, t, y):
        ynew = self._get_variables_modified(y)
        return self.event_functions(t, ynew)

    def run_event_action(self, time_, event_id):
        modified_variables = self.event_actions(time_, self._get_variables(), event_id)
        modified_mask = (modified_variables != self._get_variables())
        for idx in np.argwhere(modified_mask):
            self._write_variables(modified_variables[idx[0]], idx[0])

    def get_next_time_event(self, t) -> tuple[int, float]:
        if len(self.time_events) == 0:
            return -1, -1
        else:
            t_event_min = -1
            event_ix_min = -1
            for event_ix, timestamps in enumerate(self.time_events):
                ix = np.searchsorted(timestamps, t, 'left')
                if ix > len(timestamps) - 1:
                    continue
                else:
                    t_event = timestamps[ix]
                    if t_event_min < 0:
                        t_event_min = t_event
                        event_ix_min = event_ix
                    if t_event < t_event_min:
                        t_event_min = t_event
                        event_ix_min = event_ix

            return event_ix_min, t_event_min

    def run_time_event_action(self, t, t_event_ix):

        modified_variables = self.time_event_actions(t, self._get_variables(), t_event_ix)
        modified_mask = (modified_variables != self._get_variables())
        for idx in np.argwhere(modified_mask):
            self._write_variables(modified_variables[idx[0]], idx[0])

    def _get_variables_modified(self, y_):
        old_states = self.get_states()
        self.set_states(y_)

        vars = self._get_variables().copy()
        self.set_states(old_states)
        return vars

    def get_event_directions(self) -> np.array:
        return self.event_directions


def generate_numerous_engine_solver_interface(model: Model, nm: CompiledModel,
                                              events: tuple[CPUDispatcher, np.ndarray, CPUDispatcher],
                                              time_events: tuple[np.ndarray, CPUDispatcher],
                                              jit=True):

    NumerousEngineModelInterface_ = jithelper(NumerousEngineModelInterface, jit=jit)

    class NumerousEngineSolverInterface(SolverInterface):
        def __init__(self, interface: [NumerousEngineModelInterface, CPUDispatcher], model: Model, nm: CompiledModel):
            super(NumerousEngineSolverInterface, self).__init__(modelinterface=interface)
            self._model = model
            self._nm = nm

        def handle_solve_event(self, event_id: SolveEvent, t: float):
            if event_id == SolveEvent.Historian:
                self._model.create_historian_df()
                self._nm.historian_reinit()
            elif event_id == SolveEvent.ExternalDataUpdate:
                self._load_external_data(t)
            elif event_id == SolveEvent.HistorianAndExternalUpdate:
                self._model.create_historian_df()
                self._nm.historian_reinit()
                self._load_external_data(t)

        def _load_external_data(self, t):
            is_external_data = self._model.external_mappings.load_new_external_data_batch(t)
            external_mappings_numpy = self._model.external_mappings.external_mappings_numpy
            external_mappings_time = self._model.external_mappings.external_mappings_time
            max_external_t = self._model.external_mappings.t_max
            min_external_t = self._model.external_mappings.t_min

            if t > max_external_t:
                raise ValueError(f"No more external data at t={t} (t_max={max_external_t}")
            self._nm.is_external_data = is_external_data
            self._nm.update_external_data(external_mappings_numpy, external_mappings_time, max_external_t,
                                          min_external_t)

    return NumerousEngineSolverInterface(interface=NumerousEngineModelInterface_(nm=nm,
                                                                                 event_functions=events[0],
                                                                                 event_directions=events[1],
                                                                                 event_actions=events[2],
                                                                                 time_events=time_events[0],
                                                                                 time_event_actions=time_events[1]),
                                         model=model, nm=nm)
