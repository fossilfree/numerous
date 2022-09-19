import numpy as np
from abc import ABC
from .solve_states import SolveEvent


class ModelInterface():

    def get_deriv(self, t: float) -> np.ascontiguousarray:
        """
        Function to return derivatives of state-space model. Must be implemented by user
        :param t: time
        :return: derivatives as array
        """
        raise NotImplementedError

    def get_states(self) -> np.array:
        """
        Function to get states and return to solver. Must be implemented by user
        :return: array of states
        """
        raise NotImplementedError

    def set_states(self, y: np.array) -> None:
        """
        Function called by solver to overwrite states. Must be implemented by user.
        :param y:
        :return: None
        """
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
        :return: list of values for the all events connected to the model.
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

    def get_event_directions(self) -> np.array:
        """
        Function that returns the event directions. Must be implemented by user if using event.
        :return: list of directions, the length of the array of events
        """

        return np.array([0])

    def run_time_event_action(self, t: float, idx: int) -> None:
        return

    def get_jacobian(self, t):
        """
        Method for calculating the jacobian matrix.
        :param t:
        :return:
        """
        raise NotImplementedError


class SolverInterface(ABC):
    def __init__(self, modelinterface: ModelInterface):
        self.model: ModelInterface = modelinterface

    def handle_solve_event(self, event_id: SolveEvent, t: float):

        if event_id == SolveEvent.Historian:
            pass
        elif event_id == SolveEvent.ExternalDataUpdate:
            pass
        elif event_id == SolveEvent.HistorianAndExternalUpdate:
            pass




