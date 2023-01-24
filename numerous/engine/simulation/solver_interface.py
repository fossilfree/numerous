import numpy as np
import numpy.typing as npt
from numerous.engine.model.compiled_model import CompiledModel
from numerous.engine.model import Model
from numerous.solver.interface import Interface, EventHandler, Model as SolverModel, SolveEvent

@SolverModel
class NumerousEngineModel:
    def __init__(self, numba_model, event_functions, event_directions, event_actions, time_stamped_events,
                 periodic_time_events, time_event_actions):
        self.numba_model = numba_model
        self.event_functions = event_functions
        self.event_directions = event_directions
        self.event_actions = event_actions
        self.time_stamped_events = time_stamped_events
        self.periodic_time_events = periodic_time_events
        self.time_event_actions = time_event_actions


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

@SolverModel.interface(model=NumerousEngineModel)
class NumerousEngineModelInterface(Interface):

    """
    The numerous engine solver interface. Contains the numerous engine model, which holds the numba model, and other
    methods necessary for handling events and time-events inside Numerous engine.
    """

    def __init__(self, model: NumerousEngineModel):
        self.model: NumerousEngineModel = model
        self.last_time_event_idx = np.zeros(len(model.time_stamped_events) + len(model.periodic_time_events),
                                       dtype='int64')

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

    def init_solver(self, t: float, y: np.array) -> None:
        self.last_time_event_idx = np.zeros(len(self.model.time_stamped_events) + len(self.model.periodic_time_events),
                                            dtype='int64')

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

    def _get_next_timestamped_event(self, t, t_events) -> np.array:
        for event_ix, time_events in self.model.time_stamped_events:
            if event_ix < 0:
                continue
            possible_time_events = [t for t in time_events[self.last_time_event_idx[event_ix]:] if t >= 0]
            ix = np.searchsorted(possible_time_events, t, 'left')
            if ix > len(possible_time_events) - 1:
                continue
            else:
                t_events[event_ix] = possible_time_events[ix]
        return t_events

    def _get_next_periodic_event(self, t, t_events: np.ndarray) -> np.array:
        for event_ix, dt in self.model.periodic_time_events:
            if event_ix < 0:
                continue
            t_events[event_ix] = self.last_time_event_idx[event_ix] * dt
        return t_events

    def _possible_time_events(self, t):
        t_events = np.ones_like(self.last_time_event_idx, dtype='float') * -1.0
        t_events = self._get_next_timestamped_event(t, t_events)
        t_events = self._get_next_periodic_event(t, t_events)
        return t_events

    def get_next_time_event(self, t) -> (np.array, float):

        t_events = self._possible_time_events(t)

        possible_t_events = np.array([t_event for t_event in t_events if t_event >= 0])

        if len(possible_t_events) == 0:
            return np.array([-1]), -1.0
        first_event = np.min(possible_t_events)
        ix = np.argwhere(np.abs(first_event - t_events) < 1e-6).flatten()  # multiple events are not sorted

        return ix, first_event

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
    event_actions = model.generate_event_action_ast(model.events)
    if len(model.timestamp_events) == 0:
        model._generate_mock_timestamp_event()
    time_event_actions = model.generate_event_action_ast(model.timestamp_events)

    # Extract time events

    time_stamped_events = [(event_ix, event.timestamps) for event_ix, event in enumerate(model.timestamp_events) if
                           event.timestamps]

    time_stamped_events = numpy_fill_array(time_stamped_events)

    periodic_time_events = [(event_ix, event.periodicity) for event_ix, event in enumerate(model.timestamp_events) if
                            event.periodicity]

    if len(time_stamped_events) == 0:
        time_stamped_events = [(-1, np.array([-1.0]))]

    if len(periodic_time_events) == 0:
        periodic_time_events = [(-1, -1.0)]

    numerous_engine_model = NumerousEngineModel(model.numba_model, event_functions, event_directions, event_actions,
                                                tuple(time_stamped_events), tuple(periodic_time_events),
                                                time_event_actions)

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