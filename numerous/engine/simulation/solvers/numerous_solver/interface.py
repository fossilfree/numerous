import numpy as np
from abc import ABC
from enum import IntEnum, unique

@unique
class SolveStatus(IntEnum):
    Running = 0
    Finished = 1


@unique
class SolveEvent(IntEnum):
    NoneEvent = 0
    Historian = 1
    ExternalDataUpdate = 2
    HistorianAndExternalUpdate = 3

class ModelEvent():
    def check_event(self, t: float, y: np.array) -> float:
        return 0

class ModelInterface():

    def get_deriv(self, t: float, y: np.array) -> np.array:
        raise NotImplementedError

    def get_residual(self, t: float, yold: np.array, y: np.array, dt: float, order: int, a, af) -> np.array:
        raise NotImplementedError

    def vectorized_full_jacobian(self, t: float, y: np.array, h: float) -> np.ascontiguousarray:
        raise NotImplementedError

    def get_states(self) -> np.array:
        raise NotImplementedError

    def set_states(self, states: np.array) -> None:
        raise NotImplementedError

    def read_variables(self) -> np.array:
        raise NotImplementedError

    def write_variables(self, value: float, idx: int) -> None:
        raise NotImplementedError

    def historian_update(self, t: float) -> SolveEvent:
        pass

    def pre_step(self, t: float) -> None:
        pass

    def post_step(self, t: float) -> SolveEvent:
        pass

    def post_event(self, t: float) -> SolveEvent:
        pass

    def check_events(self):
        pass


class SolverInterface(ABC):
    def __init__(self, interface: ModelInterface):
        self._interface = interface
        self.events = []

    def handle_solve_event(self, event_id: SolveEvent, t: float):

        if event_id == SolveEvent.Historian:
            pass
        elif event_id == SolveEvent.ExternalDataUpdate:
            pass
        elif event_id == SolveEvent.HistorianAndExternalUpdate:
            pass

    def register_event_function(self, event_function: ModelEvent):
        self.events.append(event_function)


