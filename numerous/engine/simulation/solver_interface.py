import numpy as np
import numpy.typing as npt
from numba.experimental import jitclass
from numerous.engine.model.compiled_model import CompiledModel
from numerous.engine.model import Model
from numerous.engine.simulation.solvers.numerous_solver.interface import NumerousSolverInterface, \
    NumerousSolverInternalInterface, NumerousSolverExternalInterface
from numerous.engine.simulation.solvers.numerous_solver.numerous_solver import SolveEvent

class NumerousEngineExternalSolverInterface(NumerousSolverExternalInterface):
    def __init__(self, model: Model, nm: CompiledModel):
        self.model = model
        self.nm = nm

    def load_external_data(self, t):
        is_external_data = self.model.external_mappings.load_new_external_data_batch(t)
        external_mappings_numpy = self.model.external_mappings.external_mappings_numpy
        external_mappings_time = self.model.external_mappings.external_mappings_time
        max_external_t = self.model.external_mappings.t_max
        min_external_t = self.model.external_mappings.t_min

        if t > max_external_t:
            raise ValueError(f"No more external data at t={t} (t_max={max_external_t}")
        self.nm.is_external_data = is_external_data
        self.nm.update_external_data(external_mappings_numpy, external_mappings_time, max_external_t,
                                     min_external_t)

    def handle_solve_event(self, event_id: SolveEvent, t: float):
        if event_id == SolveEvent.Historian:
            self.model.create_historian_df()
            self.nm.historian_reinit()
        elif event_id == SolveEvent.ExternalDataUpdate:
            self.load_external_data(t)
        elif event_id == SolveEvent.HistorianAndExternalUpdate:
            self.model.create_historian_df()
            self.nm.historian_reinit()
            self.load_external_data(t)



class NumerousEngineInternalSolverInterface(NumerousSolverInternalInterface):
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

    def is_store_required(self) -> bool:
        return self.nm.is_store_required()

    def is_external_data_update_needed(self, t) -> bool:
        return self.nm.is_external_data_update_needed(t)

    def historian_reinit(self) -> None:
        return self.nm.historian_reinit()

    def map_external_data(self, t) -> None:
        return self.nm.map_external_data(t)

    def historian_update(self, t) -> None:
        self.nm.historian_update(t)

    def initialize(self, t) -> None:
        self.map_external_data(t)

    def post_converged_non_event_step(self, t) -> None:
        self.map_external_data(t)



def generate_numerous_engine_solver_interface(model: Model, nm: CompiledModel, jit=True):
    spec = [('nm', nm._numba_type_.class_type.instance_type)]

    def decorator(jit=True):
        def wrapper(spec):
            if jit:
                return jitclass(spec)
            else:
                pass

        return wrapper

    decorator_ = decorator(jit)

    @decorator_(spec)
    class NumerousEngineSolverInternalInterfaceWrapper(NumerousEngineInternalSolverInterface):
        pass

    class NumerousEngineSolverInterface(NumerousSolverInterface):
        def __init__(self, external_interface: NumerousEngineExternalSolverInterface,
                     internal_interface: NumerousEngineSolverInternalInterfaceWrapper):
            self.internal = internal_interface
            self.external = external_interface

    return NumerousEngineSolverInterface(external_interface=NumerousEngineExternalSolverInterface(model=model, nm=nm),
                                         internal_interface=NumerousEngineSolverInternalInterfaceWrapper(nm=nm))