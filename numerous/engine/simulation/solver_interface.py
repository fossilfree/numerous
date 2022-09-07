import numpy as np
import numpy.typing as npt
from numba.experimental import jitclass
from numerous.engine.model.compiled_model import CompiledModel
from numerous.engine.model import Model
from numerous.engine.simulation.solvers.numerous_solver.interface import SolverInterface, \
    ModelInterface
from numerous.engine.simulation.solvers.numerous_solver.numerous_solver import SolveEvent


class NumerousEngineModelInterface(ModelInterface):
    def __init__(self, nm: CompiledModel):
        self.nm = nm

    def vectorized_full_jacobian(self, t: float, y: np.array, h=1e-8) -> np.ascontiguousarray:
        return self.nm.vectorized_full_jacobian(t, y, h)

    def get_residual(self, t, yold, y, dt, order, a, af) -> np.array:
        return self.nm.get_g(t, yold, y, dt, order, a, af)

    def get_deriv(self, t, y) -> np.array:
        return self.nm.func(t, y)

    def get_states(self) -> np.array:
        return self.nm.get_states()

    def set_states(self, states: npt.ArrayLike) -> None:
        return self.nm.set_states(states)

    def read_variables(self) -> np.array:
        return self.nm.read_variables()

    def write_variables(self, value: float, idx: int) -> None:
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

    def post_event(self, t) -> SolveEvent:
        return self.historian_update(t)


def generate_numerous_engine_solver_interface(model: Model, nm: CompiledModel, jit=True):
    spec = [('nm', nm._numba_type_.class_type.instance_type)]

    def decorator(jit=True):
        def wrapper(spec):
            if jit:
                return jitclass(spec)
            else:
                def passthrough(fun):
                    return fun
                return passthrough

        return wrapper

    decorator_ = decorator(jit)

    @decorator_(spec)
    class NumerousEngineModelInterfaceWrapper(NumerousEngineModelInterface):
        pass


    class NumerousEngineSolverInterface(SolverInterface):
        def __init__(self, interface: NumerousEngineModelInterfaceWrapper, model: Model, nm: CompiledModel):
            super(NumerousEngineSolverInterface, self).__init__(interface=interface)
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

    return NumerousEngineSolverInterface(interface=NumerousEngineModelInterfaceWrapper(nm=nm), model=model, nm=nm)