import numpy as np

from abc import ABC, abstractmethod

from numerous.engine.simulation.solvers.numerous_solver.numerous_solver import SolveEvent

class NumerousSolverInternalInterface():

    def get_deriv(self, t, y) -> np.array:
        raise NotImplementedError

    def vectorized_full_jacobian(self, t, y, h) -> np.ascontiguousarray:
        raise NotImplementedError

    def get_states(self):
        raise NotImplementedError


class NumerousSolverExternalInterface(ABC):

    @abstractmethod
    def load_external_data(self, t):
        pass

    @abstractmethod
    def handle_solve_event(self, event_id: SolveEvent, t: float):
        if event_id == SolveEvent.Historian:
            pass
        elif event_id == SolveEvent.ExternalDataUpdate:
            pass
        elif event_id == SolveEvent.HistorianAndExternalUpdate:
            pass


class NumerousSolverInterface(ABC):
    def __init__(self, internal_interface: NumerousSolverInternalInterface,
                 external_interface: NumerousSolverExternalInterface):
        self.internal = internal_interface
        self.external = external_interface


