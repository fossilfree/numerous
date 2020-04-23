import copy

import time
import uuid
import numpy as np

from numba import prange
from numerous.utils.compile_decorators import basic_njit as njit
from engine.model.equation_parser import Equation_Parser

from numerous.engine.system.connector import Connector
from numerous.utils.historyDataFrame import SimpleHistoryDataFrame
from numerous.engine.scope import Scope, TemporaryScopeWrapper, ScopeVariable
from numerous.engine.simulation.simulation_callbacks import _SimulationCallback, _Event
from numerous.engine.system.subsystem import Subsystem
from numerous.engine.variables import VariableType

import operator

class ModelNamespace:

    def __init__(self, tag):
        self.tag = tag
        self.equation_dict = {}
        self.eq_variables_ids = []
        self.variables = {}


class ModelAssembler:

    @staticmethod
    def __create_scope(eq_tag, eq_methods, eq_variables, namespace, tag, variables):
        scope_id = "{0}_{1}_{2}".format(eq_tag, namespace.tag, tag, str(uuid.uuid4()))
        scope = Scope(scope_id)
        for variable in eq_variables:
            scope.add_variable(variable)
            variable.bound_equation_methods = eq_methods
            variable.parent_scope_id = scope_id
            # Needed for updating states after solve run
            if variable.type.value == VariableType.STATE.value:
                variable.associated_state_scope.append(scope_id)
            variables.update({variable.id: variable})
        return scope

    @staticmethod
    def t_1(input_namespace):
        scope_select = {}
        variables = {}
        equation_dict = {}
        tag, namespaces = input_namespace
        for namespace in namespaces:
            for i, (eq_tag, eq_methods) in enumerate(namespace.equation_dict.items()):
                scope = ModelAssembler.__create_scope(eq_tag, eq_methods,
                                                      map(namespace.variables.get, namespace.eq_variables_ids[i]),
                                                      namespace, tag, variables)
                scope_select.update({scope.id: scope})
                equation_dict.update({scope.id: eq_methods})

        return variables, scope_select, equation_dict


class Model:
    """
     The model object traverses the system to collect all information needed to pass to the solver
     for computation – the model also back-propagates the numerical results from the solver into the system,
     so they can be accessed as variable values there.
    """

    def __init__(self, system=None, historian=None, assemble=True, validate=False):

        self.system = system
        self.events = {}
        self.historian = historian or SimpleHistoryDataFrame()
        self.callbacks = [self.historian.callback]
        self.derivatives = {}
        self.model_items = {}
        self.state_history = {}
        self.synchronized_scope = {}
        self.compiled_eq = []
        self.flat_scope_idx = None
        self.flat_scope_idx_from = None

        self.global_variables_tags = ['time']
        self.global_vars = np.array([0])

        self.equation_dict = {}
        self.scope_variables = {}
        self.variables = {}
        self.flat_variables = {}
        self.path_variables = {}
        self.path_scope_variables = {}
        self.states = {}
        self.period = 1
        self.mapping_from = []
        self.mapping_to = []
        self.sum_mapping_from = []
        self.sum_mapping_to = []
        self.scope_variables_flat = []
        self.states_idx = []
        self.derivatives_idx = []
        self.scope_to_variables_idx = []

        self.info = {}
        if assemble:
            self.assemble()

        if validate:
            self.validate()


    def sychronize_scope(self):
        """
        Synchronize the values between ScopeVariables and SystemVariables
        """
        for scope_var in self.scope_variables.values():
            if scope_var.get_value() != self.variables[scope_var.id].get_value():
                scope_var.value = self.variables[scope_var.id].get_value()

    def __restore_state(self):
        for key, value in self.historian.get_last_state():
            self.scope_variables[key] = value

    def sychronize_scope(self):
        """
        Synchronize the values between ScopeVariables and SystemVariables
        """
        self.scope_variables_flat = self.flat_variables[self.scope_to_variables_idx.flatten()]

    def _get_initial_scope_copy(self):
        return TemporaryScopeWrapper(self.scope_variables_flat.copy(), self.states_idx, self.derivatives_idx)

    def __add_item(self, item):
        model_namespaces = []
        if item.id in self.model_items:
            return model_namespaces

        if item.callbacks:
            self.callbacks.append(item.callbacks)

        self.model_items.update({item.id: item})
        model_namespaces.append((item.id, self.create_model_namespaces(item)))
        if isinstance(item, Connector):
            for binded_item in item.get_binded_items():
                model_namespaces.extend(self.__add_item(binded_item))
        if isinstance(item, Subsystem):
            for registered_item in item.registered_items.values():
                model_namespaces.extend(self.__add_item(registered_item))
        return model_namespaces

    def assemble(self):
        """
        Assembles the model.
        """
        assemble_start = time.time()

        # 1. Create list of model namespaces
        model_namespaces = [_ns
                            for item in self.system.registered_items.values()
                            for _ns in self.__add_item(item)]

        # 2. Compute dictionaries
        # equation_dict <scope_id, [Callable]>
        # synchronized_scope <scope_id, Scope>
        # scope_variables <variable_id, Variable>
        for variables, scope_select, equation_dict in map(ModelAssembler.t_1, model_namespaces):

            self.equation_dict.update(equation_dict)
            self.synchronized_scope.update(scope_select)
            self.scope_variables.update(variables)

        equation_parser  = Equation_Parser()
        self.compiled_eq, self.compiled_eq_idxs = equation_parser.parse(self)


        # 4. Create self.states_idx and self.derivatives_idx
        # Fixes each variable's var_idx (position), updates variables[].idx_in_scope
        self.scope_variables_flat = np.fromiter(
            map(operator.attrgetter('value'), self.scope_variables.values()),
            np.float64)
        for var_idx, var in enumerate(self.scope_variables.values()):
            var.position = var_idx
            self.variables[var.id].idx_in_scope.append(var_idx)
        _fst = operator.itemgetter(0)
        _snd_is_derivative = lambda var: var[1].type == VariableType.DERIVATIVE
        _snd_is_state = lambda var: var[1].type == VariableType.STATE
        self.states_idx = np.fromiter(map(_fst, filter(
            _snd_is_state, enumerate(self.scope_variables.values()))),
                                      np.int64)
        self.derivatives_idx = np.fromiter(map(_fst, filter(
            _snd_is_derivative, enumerate(self.scope_variables.values()))),
                                           np.int64)

        def __get_mapping__idx(variable):
            if variable.mapping:
                return __get_mapping__idx(variable.mapping)
            else:
                return variable.idx_in_scope[0]

        # Two lookup arrays: <(scope_idx, var_idx_in_scope)>, var_idx>
        non_flat_scope_idx_from = [[] for _ in range(len(self.synchronized_scope))]
        non_flat_scope_idx = [[] for _ in range(len(self.synchronized_scope))]
        sum_idx = []
        sum_mapped = []
        sum_mapped_idx = []

        for scope_idx, scope in enumerate(self.synchronized_scope.values()):
            for scope_var_idx, var in enumerate(scope.variables.values()):
                _from = __get_mapping__idx(self.variables[var.mapping_id]) \
                    if var.mapping_id else var.position
                non_flat_scope_idx_from[scope_idx].append(_from)
                non_flat_scope_idx[scope_idx].append(var.position)
                if not var.mapping_id and var.sum_mapping_ids:
                    sum_idx.append(self.variables[var.id].idx_in_scope[0])
                    start_idx = len(sum_mapped)
                    sum_mapped += [self.variables[_var_id].idx_in_scope[0]
                                   for _var_id in var.sum_mapping_ids]
                    end_idx = len(sum_mapped)
                    sum_mapped_idx.append([start_idx, end_idx])

        ##indication of ending
        # non_flat_scope_idx_from.append(np.array([-1]))
        # non_flat_scope_idx.append(np.array([-1]))

        # TODO @Artem: document these
        self.non_flat_scope_idx_from = np.array(non_flat_scope_idx_from)
        self.non_flat_scope_idx = np.array(non_flat_scope_idx)
        self.flat_scope_idx_from = np.array([x for xs in self.non_flat_scope_idx_from for x in xs])
        self.flat_scope_idx = np.array([x for xs in self.non_flat_scope_idx for x in xs])
        self.sum_idx = np.array(sum_idx)
        self.sum_mapped = np.array(sum_mapped)
        self.sum_mapped_idx = np.array(sum_mapped_idx)

        # eq_idx -> #variables used. Can this be deduced earlier in a more elegant way?
        self.num_vars_per_eq = np.fromiter(map(len, self.non_flat_scope_idx), np.int64)[
            np.unique(self.compiled_eq_idxs, return_index=True)[1]]
        # eq_idx -> # equation instances
        self.num_uses_per_eq = np.unique(self.compiled_eq_idxs, return_counts=True)[1]

        # 6. Compute self.path_variables
        for variable in self.variables.values():
            for path in variable.path.path[self.system.id]:
                self.path_variables.update({path: variable})

        self.flat_variables = np.array([x.value for x in self.variables.values()])
        self.flat_variables_ids = [x.id for x in self.variables.values()]
        self.scope_to_variables_idx = np.array([x.idx_in_scope for x in self.variables.values()])

        _index_helper_counter = np.zeros(len(self.compiled_eq), int)
        self.index_helper = np.empty(len(self.synchronized_scope), int)
        # self.scope_variables_2d = list(map(np.empty, zip(self.num_uses_per_eq, self.num_vars_per_eq)))
        # numba needs to type this list
        self.scope_variables_2d = List()
        any(map(self.scope_variables_2d.append, map(np.empty,
            zip(self.num_uses_per_eq, self.num_vars_per_eq))))
        for scope_idx, (_flat_scope_idx_from, eq_idx) in enumerate(
                zip(self.non_flat_scope_idx_from, self.compiled_eq_idxs)):
            _idx = _index_helper_counter[eq_idx]
            _index_helper_counter[eq_idx] += 1
            self.index_helper[scope_idx] = _idx
            self.scope_variables_2d[eq_idx][_idx] = self.scope_variables_flat[_flat_scope_idx_from]

        # This can be done more efficiently using two num_scopes-sized view of a (num_scopes+1)-sized array
        flat_scope_idx_from_lengths = list(map(len, non_flat_scope_idx_from))
        self.flat_scope_idx_from_idx_2 = np.cumsum(flat_scope_idx_from_lengths)
        self.flat_scope_idx_from_idx_1 = np.hstack([[0], self.flat_scope_idx_from_idx_2[:-1]])

        flat_scope_idx_lengths = list(map(len, non_flat_scope_idx))
        self.flat_scope_idx_idx_2 = np.cumsum(flat_scope_idx_lengths)
        self.flat_scope_idx_idx_1 = np.hstack([[0], self.flat_scope_idx_idx_2[:-1]])

        assemble_finish = time.time()
        self.info.update({"Assemble time": assemble_finish - assemble_start})
        self.info.update({"Number of items": len(self.model_items)})
        self.info.update({"Number of variables": len(self.scope_variables)})
        self.info.update({"Number of equation scopes": len(self.equation_dict)})
        self.info.update({"Number of equations": len(self.compiled_eq)})
        self.info.update({"Solver": {}})

    def get_states(self):
        """

        Returns
        -------
        states : list of states
            list of all states.
        """
        return self.scope_variables[self.states_idx]


    def update_states(self,y):
        self.scope_variables[self.states_idx] = y

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

    def __create_scope_mappings(self):
        for scope in self.synchronized_scope.values():
            for var in scope.variables.values():
                if var.mapping_id:
                    var.mapping=self.scope_variables[var.mapping_id]
                if var.sum_mapping_id:
                    var.sum_mapping=self.scope_variables[var.sum_mapping_id]

    def restore_state(self, timestep=-1):
        """

        Parameters
        ----------
        timestep : time
            timestep that should be restored in the model. Default last known state is restored.

        Restores last saved state from the historian.
        """
        last_states = self.historian.get_last_state()
        r1 = []
        for state_name in last_states:
            if state_name in self.path_variables:
                if self.path_variables[state_name].type.value not in [VariableType.CONSTANT.value]:
                    self.path_variables[state_name].value = list(last_states[state_name].values())[0]
                if self.path_variables[state_name].type.value is VariableType.STATE.value:
                    r1.append(list(last_states[state_name].values())[0])
        self.scope_variables_flat[self.states_idx] = np.array(r1)

    @property
    def states_as_vector(self):
        """
        Returns current states values.

        Returns
        -------
        state_values : array of state values

        """
        if self.states_idx.size == 0:
            return np.array([])
        else:
            return self.scope_variables_flat[self.states_idx]

    def get_variable_path(self, id, item):
        for (variable, namespace) in item.get_variables():
            if variable.id == id:
                return "{0}.{1}".format(namespace.tag, variable.tag)
        if hasattr(item, 'registered_items'):
            for registered_item in item.registered_items.values():
                result = self.get_variable_path(id, registered_item)
                if result:
                    return "{0}.{1}".format(registered_item.tag, result)
        return ""

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
        self.scope_variables[variable_name].alias = alias

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

    def create_model_namespaces(self, item):
        namespaces_list = []
        for namespace in item.registered_namespaces.values():
            model_namespace = ModelNamespace(namespace.tag)
            equation_dict = {}
            eq_variables_ids = []
            for eq in namespace.associated_equations.values():
                equations = []
                ids = []
                for equation in eq.equations:
                    equations.append(equation)
                for vardesc in eq.variables_descriptions:
                    variable = namespace.get_variable(vardesc.tag)
                    self.variables.update({variable.id: variable})
                    ids.append(variable.id)
                equation_dict.update({eq.tag: equations})
                eq_variables_ids.append(ids)
            model_namespace.equation_dict = equation_dict
            model_namespace.eq_variables_ids = eq_variables_ids
            model_namespace.variables = {v.id: ScopeVariable(v) for v in namespace.variables.shadow_dict.values()}
            namespaces_list.append(model_namespace)
        return namespaces_list

    def update_model_from_scope(self, t_scope):
        '''
        Reads t_scope.flat_var, and converts flat variables to non-flat variables.
        Called by __end_step (called by Simulation.solve() ) and steady_state_solver.solve()
        Called every step.
        '''
        self.flat_variables = t_scope.flat_var[self.scope_to_variables_idx].sum(1)
        for var_id, flat_var in zip(self.flat_variables_ids, self.flat_variables):
            self.variables[var_id].value = flat_var


    def get_diff_(self):
        # Method that returns the differentiation function
        obj = self


        @njit
        def compute_eq(array_2d):
            for eq_idx in range(self.eq_count):
                obj.model.compiled_eq[eq_idx](array_2d[eq_idx])

        @njit
        def compute():
            if obj.sum_mapping:
                sum_mappings(obj.model.sum_idx, obj.model.sum_mapped_idx, obj.t_scope.flat_var,
                             obj.model.sum_mapped)
            mapping_ = True
            b1 = np.copy(obj.t_scope.flat_var)
            while mapping_:
                mapping_from(obj.model.compiled_eq_idxs, obj.model.index_helper, obj.model.scope_variables_2d,
                             obj.model.length, obj.t_scope.flat_var, obj.model.flat_scope_idx_from,
                             obj.model.flat_scope_idx_from_idx_1, obj.model.flat_scope_idx_from_idx_2)

                obj.compute_eq(obj.model.scope_variables_2d)

                mapping_to(obj.model.compiled_eq_idxs, obj.t_scope.flat_var, obj.model.flat_scope_idx,
                           obj.model.scope_variables_2d,
                           obj.model.index_helper, obj.model.length,
                           obj.model.flat_scope_idx_idx_1, obj.model.flat_scope_idx_idx_2)

                if obj.sum_mapping:
                    sum_mappings(obj.model.sum_idx, obj.model.sum_mapped_idx, obj.t_scope.flat_var,
                                 obj.model.sum_mapped)

                mapping_ = not np.allclose(b1, obj.t_scope.flat_var)
                b1 = np.copy(obj.t_scope.flat_var)

        @njit
        def __func(_t, y):

            obj.info["Number of Equation Calls"] += 1

            obj.t_scope.update_states(y)
            obj.model.global_vars[0] = _t

            obj.compute()

            return obj.t_scope.get_derivatives()

        return __func

@njit
def mapping_to(compiled_eq_idxs, flat_var, flat_scope_idx, scope_variables_2d, index_helper, length, id1, id2):
    for i in prange(compiled_eq_idxs.shape[0]):
        eq_idx = compiled_eq_idxs[i]
        flat_var[flat_scope_idx[id1[i]:id2[i]]] = \
            scope_variables_2d[eq_idx][index_helper[i]][:length[i]]

@njit
def mapping_from(compiled_eq_idxs, index_helper, scope_variables_2d, length, flat_var, flat_scope_idx_from, id1,
                 id2):
    for i in prange(compiled_eq_idxs.shape[0]):
        eq_idx = compiled_eq_idxs[i]
        scope_variables_2d[eq_idx][index_helper[i]][:length[i]] \
            = flat_var[flat_scope_idx_from[id1[i]:id2[i]]]

@njit
def sum_mappings(sum_idx, sum_mapped_idx, flat_var, sum_mapped):
    for i in prange(sum_idx.shape[0]):
        idx = sum_idx[i]
        slice_ = sum_mapped_idx[i]
        flat_var[idx] = np.sum(flat_var[sum_mapped[slice_[0]:slice_[1]]])