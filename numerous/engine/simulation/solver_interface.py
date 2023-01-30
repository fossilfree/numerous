import numpy as np
import numpy.typing as npt

from typing import Union

from numerous.engine.model.compiled_model import CompiledModel
from numerous.engine.model import Model
from numerous.engine.numerous_event import TimestampEvent, NumerousEvent, StateEvent

from numerous.solver import model, interface, Model as SolverModel, Interface, StateEvent as SolverStateEvent, \
    PeriodicTimeEvent as SolverPeriodicTimeEvent, TimestampedEvent as SolverTimestampedEvent, event
from numerous.solver.events import Event as SolverEvent
from numerous.solver.handlers import EventHandler
from numerous.solver.solve_states import SolveEvent

@model
class NumerousEngineModel(SolverModel):
    def __init__(self, numba_model, event_functions, event_directions, event_actions, time_stamped_events,
                 periodic_time_events, time_event_actions, time_events):
        self.numba_model = numba_model
        self.event_functions = event_functions
        self.event_directions = event_directions
        self.event_actions = event_actions
        self.time_stamped_events = time_stamped_events
        self.periodic_time_events = periodic_time_events
        self.time_event_actions = time_event_actions
        self.time_events = time_events


class NumerousEngineEventHandler(EventHandler):
    def __init__(self, model: Model, nm: CompiledModel):
        super(NumerousEngineEventHandler, self).__init__()
        self._model = model
        self._nm = nm

    def handle_solve_event(self, interface, event_id: SolveEvent, t: float):
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

@interface
class NumerousEngineModelInterface(Interface):

    """
    The numerous engine solver interface. Contains the numerous engine model, which holds the numba model, and other
    methods necessary for handling events and time-events inside Numerous engine.
    """
    model: NumerousEngineModel

    def get_deriv(self, t: float, y: np.array) -> np.ascontiguousarray:
        y = self.get_states()
        return self.model.numba_model.func(t, y)

    def get_states(self) -> np.array:
        return self.model.numba_model.get_states()

    def set_states(self, states: npt.ArrayLike) -> None:
        return self.model.numba_model.set_states(states)

    def _get_variables(self) -> np.array:
        return self.model.numba_model.read_variables()

    def _write_variables(self, value: float, idx: int) -> None:
        self.model.numba_model.write_variables(value, idx)

    def _is_store_required(self) -> bool:
        return self.model.numba_model.is_store_required()

    def _is_external_data_update_needed(self, t) -> bool:
        return self.model.numba_model.is_external_data_update_needed(t)

    def _map_external_data(self, t) -> None:
        return self.model.numba_model.map_external_data(t)

    def historian_update(self, t: float, y: np.array) -> SolveEvent:
        self.model.numba_model.historian_update(t)
        if self._is_store_required():
            return SolveEvent.Historian
        else:
            return SolveEvent.NoneEvent

    def pre_step(self, t: float, y: np.array) -> None:
        self._map_external_data(t)

    def post_step(self, t: float, y: np.array) -> SolveEvent:
        """
        Check if solver needs to break internal loop to update external historian, or input data, or both
        :param t: time
        :param y: states
        :type y: :class:`np.ndarray`
        :return: :class:`~numerous.solver.interface.SolveEvent`
        """
        self.model.numba_model.run_post_step()
        self._map_external_data(t)

        if self._is_store_required() and not self._is_external_data_update_needed(t):
            return SolveEvent.Historian
        elif not self._is_store_required() and self._is_external_data_update_needed(t):
            return SolveEvent.ExternalDataUpdate
        elif self._is_store_required() and self._is_external_data_update_needed(t):
            return SolveEvent.HistorianAndExternalUpdate
        else:
            return SolveEvent.NoneEvent

    def post_event(self, t: float, y: np.array) -> SolveEvent:
        """
        After an event, store the results inside internal historian

        :param t: time
        :type t: float
        :param y: states
        :type y: :class:`np.ndarray`
        :return: the result of calling :meth:`~engine.simulation.solver_interface.historian_update`
        :rtype: :class:`~numerous.solver.interface.SolveEvent`
        """
        return self.historian_update(t, y)

    def post_time_event(self, t: float, y: np.array) -> SolveEvent:
        """
        For numerous-engine, this method saves the solution after a time-event, and continues the solver internal loop

        :param t: time
        :type t: float
        :param y: states
        :type y: :class:`np.ndarray`
        :return: a NoneEvent to continue the solver
        :rtype: :class:`~numerous.solver.interface.SolveEvent`
        """

        self.historian_update(t, y)
        return SolveEvent.NoneEvent

    def get_event_results(self, t, y):
        ynew = self._get_variables_modified(y)
        return self.model.event_functions(t, ynew)

    def run_event_action(self, t_event: float, y: np.array, event_idx: int) -> np.array:
        """
        Runs the event action with event_idx. Numerous engine works with variables (states+derivatives+parameters), and
        the function is optimized to reduce the necessary update of the variables inside numerous engine by applying a
        mask.

        :param t_event: time of the event
        :type t_event: float
        :param y: states
        :type y: :class:`np.ndarray`
        :param event_idx: event action function index
        :type event_idx: int
        :return: updated states
        :rtype: :class:`np.ndarray`
        """
        modified_variables = self.model.event_actions(t_event, self._get_variables(), event_idx)
        states = modified_variables[self.model.numba_model.state_idx]
        variables = np.delete(modified_variables, self.model.numba_model.state_idx)
        original_variables = np.delete(self._get_variables(), self.model.numba_model.state_idx)
        modified_mask = (variables != original_variables)
        for idx in np.argwhere(modified_mask):
            self._write_variables(modified_variables[idx[0]], idx[0])

        return states

    def run_time_event_action(self, t: float, y: np.array, event_idx: int) -> np.array:
        """
        Runs the time event action with event_idx. Numerous engine works with variables (states+derivatives+parameters),
        and the function is optimized to reduce the necessary update of the variables inside numerous engine by applying
        a mask.

        :param t: time of the event
        :type t: float
        :param y: states
        :type y: :class:`np.ndarray`
        :param event_idx: event action function index
        :type event_idx: int
        :return: updated states
        :rtype: :class:`np.ndarray`
        """
        if self.model.time_events[event_idx]["is_external"]:
            return self.model.numba_model.state_idx

        self.last_time_event_idx[event_idx] += 1
        modified_variables = self.model.time_event_actions(t, self._get_variables(), event_idx)
        states = modified_variables[self.model.numba_model.state_idx]
        variables = np.delete(modified_variables, self.model.numba_model.state_idx)
        original_variables = np.delete(self._get_variables(), self.model.numba_model.state_idx)
        modified_mask = (variables != original_variables)
        for idx in np.argwhere(modified_mask):
            self._write_variables(modified_variables[idx[0]], idx[0])

        return states

    def _get_variables_modified(self, y_):
        old_states = self.get_states()
        self.set_states(y_)

        vars = self._get_variables().copy()
        self.set_states(old_states)
        return vars

    def get_event_directions(self) -> np.array:
        return self.model.event_directions

def generate_numerous_engine_solver_model(model: Model) -> (NumerousEngineModel, NumerousEngineEventHandler):
    """
    Method to generate the numerous-engine solver-model and its event handler
    :param model: The numerous engine model object
    :type model: :class:`~engine.model.Model`
    :return: tuple with numerousengine model and eventhandler
    :rtype: tuple(:class:`~engine.simulation.solver_interface.NumerousEngineModel`,
        :class:`~engine.simulation.solver_interface.NumerousEngineEventHandler`
    """

    event_functions, event_directions = model.generate_event_condition_ast()
    event_actions, event_actions_external = model.generate_event_action_ast(model.events)
    time_event_actions, time_event_actions_external = model.generate_event_action_ast(model.timestamp_events)

    # Extract time events

    time_stamped_events = [(event_ix, event.timestamps, event.is_external) for event_ix, event in
                           enumerate(model.timestamp_events) if event.timestamps]

    time_stamped_events = numpy_fill_array(time_stamped_events)

    periodic_time_events = [(event_ix, event.periodicity, event.is_external) for event_ix, event in
                            enumerate(model.timestamp_events) if event.periodicity]

    if len(time_stamped_events) == 0:
        time_stamped_events = [(-1, np.array([-1.0]), False)]

    if len(periodic_time_events) == 0:
        periodic_time_events = [(-1, -1.0, False)]

    time_events = {event_idx: is_external for (event_idx, _, is_external) in time_stamped_events}
    time_events.update({event_idx: is_external for (event_idx, _, is_external) in periodic_time_events})

    numerous_engine_model = NumerousEngineModel(model.numba_model, event_functions, event_directions, event_actions,
                                                time_event_actions, event_actions_external,
                                                time_event_actions_external, time_events)

    numerous_event_handler = NumerousEngineEventHandler(model, model.numba_model)

    return numerous_engine_model, numerous_event_handler


def numpy_fill_array(data: list[tuple]) -> list[tuple]:
    if len(data) == 0:
        return data
    data_ = [d[1] for d in data]
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data_])

    # Mask of valid places in each row
    mask = np.arange(max(lens)) < lens[:, None]

    # Setup output array and put elements from data into masked positions
    out = np.ones(mask.shape, dtype='float') * -1
    out[mask] = np.concatenate(data_)
    data_out = data
    for ix, (event_ix, _) in enumerate(data):
        data_out[ix] = (event_ix, out[ix])

    return data_out

def wrap_events(event_: Union[TimestampEvent,StateEvent], ix: int):

    def _new_time_event(BaseClass) -> type(SolverEvent):
        @event
        class NumerousTimeEvent(BaseClass):
            def run_event_action(self, interface: NumerousEngineModelInterface, t: float, y: np.array) -> np.array:
                vars = interface._get_variables()
                modified_variables = interface.model.periodic_time_events[ix](t, vars)
                states = modified_variables[interface.model.numba_model.state_idx]
                variables = np.delete(modified_variables, interface.model.numba_model.state_idx)
                original_variables = np.delete(interface._get_variables(), interface.model.numba_model.state_idx)
                modified_mask = (variables != original_variables)
                for idx in np.argwhere(modified_mask):
                    interface._write_variables(modified_variables[idx[0]], idx[0])
                return states

        return NumerousTimeEvent

    class NumerousStateEvent(SolverStateEvent):
        def run_event_action(self, interface: NumerousEngineModelInterface, t: float, y: np.array) -> np.array:
            modified_variables = interface.model.event_actions(t, interface._get_variables(), ix)
            states = modified_variables[interface.model.numba_model.state_idx]
            variables = np.delete(modified_variables, interface.model.numba_model.state_idx)
            original_variables = np.delete(interface._get_variables(), interface.model.numba_model.state_idx)
            modified_mask = (variables != original_variables)
            for idx in np.argwhere(modified_mask):
                interface._write_variables(modified_variables[idx[0]], idx[0])

            return states

        def get_event_results(self, interface: NumerousEngineModelInterface, t: float, y: np.array) -> float:
            return interface.model.event_functions(t, y)[ix]

        def get_event_directions(self, interface: NumerousEngineModelInterface) -> int:
            return interface.model.event_directions[ix]


    if type(event_) == TimestampEvent:
        if event_.periodicity:
            baseclass = SolverPeriodicTimeEvent
            periodic_event: type(SolverPeriodicTimeEvent) = _new_time_event(baseclass)
            return periodic_event(id=event_.key, periodic=event_.periodicity, is_external=event_.is_external)
        else:
            baseclass = SolverTimestampedEvent
            periodic_event: type(SolverPeriodicTimeEvent) = _new_time_event(baseclass)
            return periodic_event(id=event_.key, timestamps = event_.timestamps)
    elif type(event_) == StateEvent:
        return NumerousStateEvent(id=event_.key, is_external=event_.is_external)