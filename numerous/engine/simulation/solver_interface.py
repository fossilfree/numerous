import numpy as np
import numpy.typing as npt

from typing import Union, Callable

from numerous.engine.model.compiled_model import CompiledModel
from numerous.engine.model import Model
from numerous.engine.numerous_event import TimestampEvent, StateEvent

from numerous.solver import model, interface, Model as SolverModel, Interface, StateEvent as SolverStateEvent, \
    PeriodicTimeEvent as SolverPeriodicTimeEvent, TimestampedEvent as SolverTimestampedEvent, event
from numerous.solver.events import Event as SolverEvent
from numerous.solver.handlers import EventHandler
from numerous.solver.solve_states import SolveEvent

@model
class NumerousEngineModel(SolverModel):
    def __init__(self, numba_model,
                 state_event_functions,
                 state_event_directions,
                 state_event_actions,
                 time_event_actions):

        self.numba_model = numba_model
        self.state_event_functions = state_event_functions
        self.state_event_directions = state_event_directions
        self.state_event_actions = state_event_actions
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

    def post_state_event(self, t: float, y: np.array, event_id: str) -> SolveEvent:
        """
        After an event, store the results inside internal historian


        :param t: time
        :type t: float
        :param y: current solver and model states in numpy format
        :type y: :class:`numpy.ndarray`
        :param event_id: the id of the state event function that was triggered
        :type event_id: str
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


def generate_numerous_engine_solver_model(model: Model) -> (NumerousEngineModel, NumerousEngineEventHandler):
    """
    Method to generate the numerous-engine solver-model and its event handler

    :param model: The numerous engine model object
    :type model: :class:`~engine.model.Model`
    :return: tuple with :class:`~engine.simulation.solver_interface.NumerousEngineModel` and
    :class:`~engine.simulation.solver_interface.NumerousEngineEventHandler`
    :rtype: tuple(:class:`~engine.simulation.solver_interface.NumerousEngineModel`,
        :class:`~engine.simulation.solver_interface.NumerousEngineEventHandler`
    """

    state_event_functions, state_event_directions = model.generate_event_condition_ast()
    state_event_actions = model.generate_event_action_ast(model.events)

    state_events = []
    event_factory = EventFactory()

    for event_ix, numerousevent in enumerate(model.events):
        state_event = event_factory.create_numerous_solver_event(numerousevent, event_ix, model)
        state_events.append(state_event)

    time_event_actions = model.generate_event_action_ast(model.timestamp_events)

    time_events = []
    for event_ix, numerousevent in enumerate(model.timestamp_events):
        time_event = event_factory.create_numerous_solver_event(numerousevent, event_ix, model)
        time_events.append(time_event)


    # Extract time events

    numerous_engine_model = NumerousEngineModel(model.numba_model,
                                                state_event_functions,
                                                state_event_directions,
                                                state_event_actions,
                                                time_event_actions)

    numerous_engine_model.add_time_events(time_events)
    numerous_engine_model.add_state_events(state_events)

    numerous_event_handler = NumerousEngineEventHandler(model, model.numba_model)

    return numerous_engine_model, numerous_event_handler


class EventFactory:
    """
    Factory class for creating Numerous Solver events from :class:`~engine.numerous_events.NumerousEvent`

    """
    def _wrap_internal_event(self, ix: int, event_type: str):
        # don't use closure on event actions as it will not compile
        assert event_type == "state" or event_type == "time", "unknown event type"

        class BaseInternalEvent(SolverEvent):
            def run_event_action(self, interface: NumerousEngineModelInterface, t: float, y: np.array) -> np.array:
                vars = interface._get_variables()
                if event_type == 'state':
                    modified_variables = interface.model.state_event_actions(t, vars, ix)
                else:
                    modified_variables = interface.model.time_event_actions(t, vars, ix)
                states = modified_variables[interface.model.numba_model.state_idx]
                variables = np.delete(modified_variables, interface.model.numba_model.state_idx)
                original_variables = np.delete(interface._get_variables(), interface.model.numba_model.state_idx)
                modified_mask = (variables != original_variables)
                for idx in np.argwhere(modified_mask):
                    interface._write_variables(modified_variables[idx[0]], idx[0])
                return states

        return BaseInternalEvent
    def _wrap_external_event(self, model: Model, external_action_function: Callable, parent_path: str):
        # Use closure to wrap external event class
        class BaseExternalEvent(SolverEvent):
            def run_event_action(self, interface: NumerousEngineModelInterface, t: float, y: np.array) -> np.array:
                model.update_local_variables()
                variables = {tag: var.value for tag, var in model.path_to_variable.items()}
                path_ = ""
                if parent_path:
                    path_ = ".".join(parent_path) + "."
                    variables = {tag.strip(path_): value for tag, value in variables.items()}
                external_action_function(t, variables)
                for tag, value in variables.items():
                    model.path_to_variable[path_ + tag].value = value
                model.update_all_variables()
                states = interface.get_states()
                return states

        return BaseExternalEvent

    def _wrap_state_event(self, ix):
        class BaseStateEvent(SolverStateEvent):
            def get_event_results(self, interface: NumerousEngineModelInterface, t: float, y: np.array) -> float:
                vars = self._get_variables_modified(interface, t, y)
                return interface.model.state_event_functions(t, vars)[ix]

            def get_event_directions(self, interface: NumerousEngineModelInterface, t: float, y: np.array) -> int:
                return interface.model.state_event_directions[ix]

            def _get_variables_modified(self, interface: NumerousEngineModelInterface, t, y_) -> np.array:
                old_states = interface.get_states()
                interface.set_states(y_)

                vars = interface.model.numba_model.read_variables().copy()
                interface.set_states(old_states)
                return vars

        return BaseStateEvent

    def create_numerous_solver_event(self, event_: Union[TimestampEvent, StateEvent], ix: int, model: Model) -> \
        Union[SolverTimestampedEvent, SolverStateEvent, SolverPeriodicTimeEvent]:
        """
        Factory method ot create a numerous solver event based on the type of
        :class:`~engine.numerous_events.NumerousEvent` and it's properties (is_external, timestamps etc.)

        :param event_: The :class:`~engine.numerous_events.NumerousEvent` derived class
        :param ix: the index of the event in :attr:`~engine.model.Model.timestamp_events` and
        :attr:`~engine.model.Model.events`
        :param model: the :class:`~engine.model.Model`
        :return:

        """
        if type(event_) == TimestampEvent:
            event_type = 'time'
            if event_.periodicity:
                baseclass = SolverPeriodicTimeEvent
                kwargs = dict(id=event_.key, period=event_.periodicity, is_external=event_.is_external)
            else:
                baseclass = SolverTimestampedEvent
                kwargs = dict(id=event_.key, timestamps=event_.timestamps, is_external=event_.is_external)

        else:
            event_type = 'state'
            baseclass = self._wrap_state_event(ix)
            kwargs = dict(id=event_.key, is_external=event_.is_external)

        new_event: type = self._new_event(ix, model, baseclass,
                                          is_external=event_.is_external,
                                          external_action_function=event_.action,
                                          parent_path=event_.parent_path, event_type=event_type)

        return new_event(**kwargs)

    def _new_event(self, ix: int, model: Model, BaseEventClass: type, is_external: bool = False,
                        external_action_function: Callable = None,
                        parent_path: str = None, event_type: str = None) \
            -> type(Union[SolverTimestampedEvent, SolverPeriodicTimeEvent]):

        ExternalBaseEvent = self._wrap_external_event(model, external_action_function, parent_path)
        InternalBaseEvent = self._wrap_internal_event(ix, event_type=event_type)

        @event
        class ExternalNumerousEvent(BaseEventClass, ExternalBaseEvent):
            pass

        @event
        class InternalNumerousEvent(BaseEventClass, InternalBaseEvent):
            pass


        return ExternalNumerousEvent if is_external else InternalNumerousEvent
