import numpy as np
from abc import ABC
from enum import IntEnum, unique
from typing import Optional

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

class ModelInterface():

    def get_deriv(self, t: float, y: np.array) -> np.array:
        """
        Function to return derivatives of state-space model. Must be implemented by user
        :param t: time
        :param y: current state array
        :return: derivatives as array
        """
        raise NotImplementedError

    def get_residual(self, t: float, yold: np.array, y: np.array, dt: float, order: int, a, af) -> np.array:
        """
        Function to get residual (lhs-rhs). Used by LM method. Can be omitted if LM method is not used.
        :param t: time
        :param yold: array of last converged states
        :param y: array of (current) states
        :param dt: time-step determined by solver
        :param order: method order (1-5)
        :param a: some parameter
        :param af: some more parameters
        :return:
        """
        raise NotImplementedError

    def vectorized_full_jacobian(self, t: float, y: np.array, h: float) -> np.ascontiguousarray:
        """
        Function to generate jacobian matrix. Used with LM method. If left out, LM method cannot run.
        :param t: time
        :param y: array of states
        :param h: an optional parameter
        :return:
        """
        raise NotImplementedError

    def get_states(self) -> np.array:
        """
        Function to get states and return to solver. Must be implemented by user
        :return: array of states
        """
        raise NotImplementedError

    def set_states(self, states: np.array) -> None:
        """
        Function called by solver to overwrite states. Must be implemented by user.
        :param states:
        :return: None
        """
        raise NotImplementedError

    def read_variables(self) -> np.array:
        """
        Function to read all variables and return to solver. Must be implmented by user
        :return: array of variables.
        """
        raise NotImplementedError

    def write_variables(self, value: float, idx: int) -> None:
        raise NotImplementedError

    def historian_update(self, t: float) -> SolveEvent:
        """

        :param t: time
        :return: SolveEvent that can be used to break solver loop for external updates
        """
        return SolveEvent.NoneEvent

    def pre_step(self, t: float) -> None:
        """
        Function called once every time solver is started, also called when solve resumes after exiting due to
                 SolveEvent
        :param t: time
        :return: None
        """
        pass

    def post_step(self, t: float) -> SolveEvent:
        """
        Function called every time step has converged, and there was no event step in between.
        :param t: time
        :return: SolveEvent that can be used to break solver loop for external updates
        """
        return SolveEvent.NoneEvent

    def post_event(self, t: float) -> SolveEvent:
        """
        Function called every time solver has converged to an event step.
        :param t: time
        :return: SolveEvent that can be used to break solver loop for external updates
        """
        return SolveEvent.NoneEvent

    def get_event_results(self, t: float, y: np.array) -> np.array:
        """
        Function called to find events. Used together with event directions to determine if an event occured.
        :param t: time
        :param y: states
        :return: list of values for the all events
        """
        return np.array([0])

    def run_event_action(self, t_event: float, event_id: int) -> None:
        """
        Function called each time an event has been found, and can be used to trigger an action. The event_id is used
        to be able to distinguish which event occured.
        :param t_event: time
        :param event_id: int
        :return: None
        """
        pass

    def get_next_time_event(self, t) -> tuple[int, float]:
        """
        Function that is called after each converged solver step. Returns a tuple which contains an index of the next
        time event action function to be trigged, and the value of the time-step when the function is triggered.
        :param t: current time
        :return: tuple of index of event function and next time. If no function is triggered, it should return -1, -1.
        """
        return -1, -1


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



