import copy
import time
import uuid
from numerous.engine.system.connector import Connector
from numerous.utils.historyDataFrame import HistoryDataFrame
from numerous.engine.scope import Scope, TemporaryScopeWrapper
from numerous.engine.simulation.simulation_callbacks import _SimulationCallback,_Event
from numerous.engine.system.subsystem import Subsystem
from numerous.engine.variables import VariableType


class Model:
    """
     The model object traverses the system to collect all information needed to pass to the solver
     for computation – the model also back-propagates the numerical results from the solver into the system,
     so they can be accessed as variable values there.
    """
    def __init__(self, system=None, historian=None, assemble=True, validate=False):

        self.system = system
        self.events = {}
        self.historian = historian if historian else HistoryDataFrame()
        self.callbacks = [self.historian.callback]
        self.derivatives = {}
        self.model_items = {}
        self.state_history = {}
        self.synchronized_scope = {}
        self.equation_dict = {}
        self.variables = {}
        self.states = {}
        self.period = 1

        self.info = {}
        if assemble:
            self.assemble()

        if validate:
            self.validate()

    def sychronize_scope(self):
        """
        Synchronize the values between ScopeVariables and SystemVariables
        """
        for scope in self.synchronized_scope.values():
            for scope_var in scope.variables_dict.values():
                if scope_var.value != scope_var.base_variable.value:
                    scope_var.value = scope_var.base_variable.value

    def __restore_state(self):
        for key, value in self.historian.get_last_state():
            self.variables[key] = value

    def __update_var_desc(self, namespace, scope, vardesc, scope_id):
        variable = namespace.get_variable(vardesc)
        scope.add_variable(variable)
        if vardesc.type.value == VariableType.STATE.value:
            variable.associated_scope.append(scope_id)
            self.states.update({variable.id: variable})
        if vardesc.type.value == VariableType.DERIVATIVE.value:
            self.derivatives.update({variable.id: variable})
        var_path = self.get_variable_path(variable.id, self.system)
        self.variables.update({var_path: variable})

    def __create_scope(self, eq, namespace, item):
        scope = Scope()
        scope_id = "{0}_{1}_{2}_{3}".format(eq.tag, namespace.tag, item.tag, str(uuid.uuid4()))
        for var_desc in eq.variables_descriptions:
            self.__update_var_desc(namespace, scope, var_desc, scope_id)
        self.synchronized_scope.update({scope_id: scope})
        return scope_id

    def _update_scope_states(self, new_state):

        for i, v1 in enumerate(self.states.values()):
            v1.value = new_state[i]

    def _get_initial_scope_copy(self):
        return TemporaryScopeWrapper(copy.copy(self.synchronized_scope), self.states)


    def __add_item(self, item):
        if item.id in self.model_items:
            return

        if item.callbacks:
            self.callbacks.append(item.callbacks)

        self.model_items.update({item.id: item})
        if isinstance(item, Connector):
            for binded_item in item.get_binded_items():
                self.__add_item(binded_item)
        if isinstance(item, Subsystem):
            for registered_item in item.registered_items.values():
                self.__add_item(registered_item)

    def assemble(self):
        """
        Assembles the model.
        """
        assemble_start = time.time()
        for item in self.system.registered_items.values():
            self.__add_item(item)

        for item in self.model_items.values():
            for namespace in item.registered_namespaces.values():
                for eq in namespace.associated_equations.values():
                    scope_id = self.__create_scope(eq, namespace, item)
                    self.equation_dict.update({scope_id: eq})

        self.__create_scope_mappings()
        assemble_finish = time.time()
        self.info.update({"Assemble time": assemble_finish - assemble_start})
        self.info.update({"Number of items": len(self.model_items)})
        self.info.update({"Number of variables": len(self.variables)})
        self.info.update({"Number of equation scopes": len(self.equation_dict)})
        self.info.update({"Solver": {}})

    def get_states(self):
        """

        Returns
        -------
        states : list of states
            list of all states.
        """
        return self.states

    def validate(self):
        """
        Checks that all bindings are fulfilled.
        """
        valid = True
        for item in self.model_items.values():
            for binding in item.bindings:
                if binding.is_bindend():
                    pass
                else:
                    valid = False
        return valid

    def search_items(self, item_tag):
        """
        Search an item in items registered in the model by a tag

        Returns
        ----------
        items : list of :class:`numerous.engine.system.Item`
            set of items with given tag
               """
        return [item for item in self.model_items.values() if item.tag == item_tag]

    def __find_scope_var(self, var_id):
        for scope in self.synchronized_scope.values():
            for scope_var in scope.variables_dict.values():
                if scope_var.id == var_id:
                    return scope_var

    def __create_scope_mappings(self):
        for scope in self.synchronized_scope.values():
            for var in scope.variables_dict.values():
                for mapping_id in var.mapping_ids:
                    var.mapping.append(self.__find_scope_var(mapping_id))

    def restore_state(self, timestep=-1):
        """

        Parameters
        ----------
        timestep : time
            timestep that should be restored in the model. Default last known state is restored.

        Restores last saved state from the historian.
        """
        last_states = self.historian.get_last_state()
        for state_name in last_states:
            if state_name in self.variables:
                if  self.variables[state_name].type.value not in [VariableType.CONSTANT.value]:
                    self.variables[state_name].value = list(last_states[state_name].values())[0]

    @property
    def states_as_vector(self):
        """
        Returns current states values.

        Returns
        -------
        state_values : list of state values
            a default namespace.

        """
        return [x.value for x in self.states.values()]

    def save_variables_schedule(self, period, filename):
        """
        Save data to file on given period.

        Parameters
        ----------
        period : timedelta
            timedelta of saving history to file

        filename : string
            Name of a file
        Returns
        -------

        """
        self.period = period

        def saver_callback(t, _):
            if t > self.period:
                self.historian.save(filename)
                self.period = t + self.period

        callback = _SimulationCallback("FileWriter")
        callback.add_callback_function(saver_callback)
        self.callbacks.append(callback)

    def get_variable_path(self, id, item):
        """
        Create a path to a given variable

        Parameters
        ----------
        id : id
            id of a variable

        item : :class:`numerous.engine.system.Item`
            Item to search

        Returns
        -------
        path: string
            path to variable

        """
        for (variable, namespace) in item.get_variables():
            if variable.id == id:
                return "{0}.{1}".format(namespace.tag, variable.tag)
        if hasattr(item, 'registered_items'):
            for registered_item in item.registered_items.values():
                result = self.get_variable_path(id, registered_item)
                if result:
                    return "{0}.{1}".format(registered_item.tag, result)
        return ""

    def add_event(self, name, event_function, callbacks=None):
        """
        Creating and adding Event callback.


        Parameters
        ----------
        name : string
            name of the event

        event_function : callable


        callbacks : list of callable
            callback associated with event

        Returns
        -------

        """
        if not callbacks:
            callbacks = []
        self.events.update({name: _Event(name, self, event_function=event_function, callbacks=callbacks)})

    def add_event_callback(self, event_name, event_callback):
        """
        Adding the callback to existing event

        Parameters
        ----------
        event_name : string
            name of the registered event

        event_callback : callable
            callback associated with event


        Returns
        -------

        """
        self.events[event_name].add_callbacks(event_callback)

    def create_alias(self, variable_name, alias):
        """

        Parameters
        ----------
        variable_name
        alias

        Returns
        -------

        """
        self.variables[variable_name].alias = alias

    def add_callback(self, name, callback_function):
        """
        Adding a callback


        Parameters
        ----------
        name : string
            name of the callback

        callback_function : callable
            callback function

        """
        self.callbacks.append(_SimulationCallback(name, callback_function))
