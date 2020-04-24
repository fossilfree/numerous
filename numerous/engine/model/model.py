import copy

import time
import uuid
import numpy as np
from numba import prange, jitclass
from numerous.utils.compile_decorators import basic_njit as njit
from engine.model.equation_parser import Equation_Parser
from numerous.engine.system.connector import Connector
from numerous.utils.historyDataFrame import SimpleHistoryDataFrame
from numerous.engine.scope import Scope, ScopeVariable
from numerous.engine.simulation.simulation_callbacks import _SimulationCallback, _Event
from numerous.engine.system.subsystem import Subsystem
from numerous.engine.variables import VariableType


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
        self.global_vars = np.array([0], dtype=np.float64)

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

    def __restore_state(self):
        for key, value in self.historian.get_last_state():
            self.scope_variables[key] = value

    def sychronize_scope(self):
        """
        Synchronize the values between ScopeVariables and SystemVariables
        """
        self.scope_variables_flat = self.flat_variables[self.scope_to_variables_idx.flatten()]


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

        equation_parser = Equation_Parser()
        self.compiled_eq, self.compiled_eq_idxs = equation_parser.parse(self)

        # 4. Create self.states_idx and self.derivatives_idx
        # Fixes each variable's var_idx (position), updates variables[].idx_in_scope
        for var_idx, variable in enumerate(self.scope_variables.values()):
            self.scope_variables_flat.append(variable.value)
            variable.position = var_idx
            self.variables[variable.id].idx_in_scope.append(var_idx)
            if variable.type.value == VariableType.STATE.value:
                self.states_idx.append(var_idx)
            elif variable.type.value == VariableType.DERIVATIVE.value:
                self.derivatives_idx.append(var_idx)

        self.states_idx = np.array(self.states_idx,dtype=np.int32)
        self.derivatives_idx = np.array(self.derivatives_idx, dtype=np.int32)

        def __get_mapping__idx(variable):
            if variable.mapping:
                return __get_mapping__idx(variable.mapping)
            else:
                return variable.idx_in_scope[0]

        # Two lookup arrays: <(scope_idx, var_idx_in_scope)>, var_idx>
        flat_scope_idx_from = [[] for _ in range(len(self.synchronized_scope))]
        flat_scope_idx = [[] for _ in range(len(self.synchronized_scope))]
        sum_idx = []
        sum_mapped = []
        sum_mapped_idx = []

        for scope_idx, scope in enumerate(self.synchronized_scope.values()):
            for scope_var_idx, var in enumerate(scope.variables.values()):
                _from = __get_mapping__idx(self.variables[var.mapping_id]) \
                    if var.mapping_id else var.position
                flat_scope_idx_from[scope_idx].append(_from)
                flat_scope_idx[scope_idx].append(var.position)
                if not var.mapping_id and var.sum_mapping_ids:
                    sum_idx.append(self.variables[var.id].idx_in_scope[0])
                    start_idx = len(sum_mapped)
                    sum_mapped += [self.variables[_var_id].idx_in_scope[0]
                                   for _var_id in var.sum_mapping_ids]
                    end_idx = len(sum_mapped)
                    sum_mapped_idx.append([start_idx, end_idx])

        # TODO @Artem: document these
        self.flat_scope_idx_from = np.array(flat_scope_idx_from)
        self.flat_scope_idx = np.array(flat_scope_idx)
        self.sum_idx = np.array(sum_idx, dtype=np.int32)
        self.sum_mapped = np.array(sum_mapped, dtype=np.float64)
        self.sum_mapped_idx = np.array(sum_mapped_idx, dtype=np.int32)

        # 6. Compute self.path_variables
        for variable in self.variables.values():
            for path in variable.path.path[self.system.id]:
                self.path_variables.update({path: variable})

        # float64 array of all variables' current value
        self.flat_variables = np.array([x.value for x in self.variables.values()])
        self.flat_variables_ids = [x.id for x in self.variables.values()]
        self.scope_to_variables_idx = np.array([np.array(x.idx_in_scope) for x in self.variables.values()])

        self.scope_variables_flat = np.array(self.scope_variables_flat, dtype=np.float64, order='F')

        eq_count = len(self.compiled_eq)

        # (eq_idx, ind_eq_access) -> scope_variable.value
        _scope_variables_2d = [[] for _ in range(eq_count)]
        self.index_helper = np.empty(len(self.synchronized_scope), int)
        max_scope_len = max(map(len, self.flat_scope_idx_from))
        for scope_id, (_flat_scope_idx_from, eq_idx) in enumerate(
                zip(self.flat_scope_idx_from, self.compiled_eq_idxs)):
            self.index_helper[scope_id] = len(_scope_variables_2d[eq_idx])
            _scope_variables_2d[eq_idx].append(self.scope_variables_flat[_flat_scope_idx_from])

        # self.index_helper: how many of the same item type do we have?
        # max_scope_len: maximum number of variables one item can have
        # not correcly sized, as np.object
        # (eq_idx, ind_of_eq_access, var_index_in_scope) -> scope_variable.value
        # Artem: are you sure "ones" is what you want for padding on axis 1?
        #        Otherwise the padding 5 lines down can be removed
        self.scope_variables_2d = np.ones([eq_count, np.max(self.index_helper) + 1, max_scope_len],dtype=np.float64)
        for eq_idx, _ind_of_eq_access in zip(self.compiled_eq_idxs, self.index_helper):
            _vals = _scope_variables_2d[eq_idx][_ind_of_eq_access]
            self.scope_variables_2d[eq_idx, _ind_of_eq_access, :len(_vals)] = _vals
            self.scope_variables_2d[eq_idx, _ind_of_eq_access, len(_vals):] = 0

        self.length = np.array(list(map(len, self.flat_scope_idx)))

        flat_scope_idx_from_lengths = list(map(len, flat_scope_idx_from))
        self.flat_scope_idx_from_idx_2 = np.cumsum(flat_scope_idx_from_lengths)
        self.flat_scope_idx_from_idx_1 = np.hstack([[0], self.flat_scope_idx_from_idx_2[:-1]])
        flat_scope_idx_from_flat = np.empty(sum(flat_scope_idx_from_lengths), int)
        for start_idx, item in zip(self.flat_scope_idx_from_idx_1, flat_scope_idx_from):
            flat_scope_idx_from_flat[start_idx:start_idx + len(item)] = item
        self.flat_scope_idx_from = flat_scope_idx_from_flat

        flat_scope_idx_lengths = list(map(len, flat_scope_idx))
        self.flat_scope_idx_idx_2 = np.cumsum(flat_scope_idx_lengths)
        self.flat_scope_idx_idx_1 = np.hstack([[0], self.flat_scope_idx_idx_2[:-1]])
        flat_scope_idx_flat = np.empty(sum(flat_scope_idx_lengths), int)
        for start_idx, item in zip(self.flat_scope_idx_idx_1, flat_scope_idx):
            flat_scope_idx_flat[start_idx:start_idx + len(item)] = item
        self.flat_scope_idx = flat_scope_idx_flat

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

    def update_states(self, y):
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
                    var.mapping = self.scope_variables[var.mapping_id]
                if var.sum_mapping_id:
                    var.sum_mapping = self.scope_variables[var.sum_mapping_id]

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

    def update_model_from_scope(self):
        '''
        Reads t_scope.flat_var, and converts flat variables to non-flat variables.
        Called by __end_step (called by Simulation.solve() ) and steady_state_solver.solve()
        Called every step.
        '''
        self.flat_variables = self.scope_variables_flat[self.scope_to_variables_idx].sum(1)
        for i, v_id in enumerate(self.flat_variables_ids):
            self.variables[v_id].value = self.flat_variables[i]

    # Method that returns the differentiation function
    def get_diff_(self):
        compiled_eq = self.compiled_eq
        eq_count =  len(compiled_eq)

        from numba import int32,float64,int64,boolean
        from numba import types
        spec = [
            ('sum_idx', int32[:]),
            ('sum_mapped_idx', int32[:]),
            ('sum_mapped', float64[:]),
            ('sum_mapping', boolean),
            ('compiled_eq_idxs', int32[:]),
            ('index_helper', int64[:]),
            ('length', int64[:]),
            ('flat_scope_idx_from', int64[:]),
            ('flat_scope_idx_from_idx_1', int64[:]),
            ('flat_scope_idx_from_idx_2', int64[:]),
            ('flat_scope_idx', int64[:]),
            ('flat_scope_idx_idx_1', int64[:]),
            ('flat_scope_idx_idx_2', int64[:]),
            ('flat_var', float64[:]),
            ('state_idx', int32[:]),
            ('deriv_idx', int32[:]),
            ('global_vars', float64[:]),
            ('scope_variables_2d', float64[:,:,:]),
        ]



        class NumbaModel:
            def __init__(self,
                         sum_idx, sum_mapped_idx,
                        sum_mapped, compiled_eq_idxs,
                        index_helper, length, flat_scope_idx_from,
                        flat_scope_idx_from_idx_1,flat_scope_idx_from_idx_2,
                        flat_scope_idx, flat_scope_idx_idx_1,flat_scope_idx_idx_2,
                        flat_scope_var, state_idx,deriv_idx, global_vars,scope_variables_2d):
                self.sum_idx = sum_idx
                self.sum_mapped_idx = sum_mapped_idx
                self.sum_mapped = sum_mapped
                self.sum_mapping = sum_idx.size != 0
                self.compiled_eq_idxs = compiled_eq_idxs
                self.index_helper = index_helper
                self.length = length
                self.flat_scope_idx_from = flat_scope_idx_from
                self.flat_scope_idx_from_idx_1 = flat_scope_idx_from_idx_1
                self.flat_scope_idx_from_idx_2 = flat_scope_idx_from_idx_2
                self.flat_scope_idx = flat_scope_idx
                self.flat_scope_idx_idx_1 = flat_scope_idx_idx_1
                self.flat_scope_idx_idx_2 = flat_scope_idx_idx_2
                self.flat_var = flat_scope_var
                self.state_idx = state_idx
                self.deriv_idx = deriv_idx
                self.global_vars = global_vars
                self.scope_variables_2d = scope_variables_2d


            def update_states(self, state_values):
                self.flat_var[self.state_idx] = state_values

            def compute_eq(self,array_2d):
                self.func0(array_2d[0])
                for eq_idx in range(eq_count):
                    compiled_eq[eq_idx](array_2d[eq_idx])

            def update_states_idx(self, state_value, idx):
                self.flat_var[idx] = state_value

            def get_derivatives(self):
                return self.flat_var[self.deriv_idx]

            def get_derivatives_idx(self, idx):
                return self.flat_var[idx]

            def compute(self):
                # if self.sum_mapping:
                #     sum_mappings(self.sum_idx, self.sum_mapped_idx,
                #                  self.flat_var,
                #                  self.sum_mapped)
                mapping_ = True
                b1 = np.copy(self.flat_var)
                while mapping_:
                    mapping_from(self.compiled_eq_idxs, self.index_helper,
                                 self.scope_variables_2d,
                                 self.length, self.flat_var,
                                 self.flat_scope_idx_from,
                                 self.flat_scope_idx_from_idx_1,
                                 self.flat_scope_idx_from_idx_2)

                    self.compute_eq(self.scope_variables_2d)

                    mapping_to(self.compiled_eq_idxs, self.flat_var,
                               self.flat_scope_idx,
                               self.scope_variables_2d,
                               self.index_helper, self.length,
                               self.flat_scope_idx_idx_1, self.flat_scope_idx_idx_2)

                    # if self.sum_mapping:
                    #     sum_mappings(self.sum_idx, self.sum_mapped_idx,
                    #                  self.flat_var,
                    #                  self.sum_mapped)

                    mapping_ = not np.allclose(b1, self.flat_var)
                    b1 = np.copy(self.flat_var)

            def func(self, _t, y):

                self.update_states(y)
                self.global_vars[0] = _t
                self.compute()

                return self.get_derivatives()

        for i,function in enumerate(self.compiled_eq):
            setattr(NumbaModel, 'func'+str(i), function)

        @jitclass(spec)
        class NumbaModel2(NumbaModel):
            pass

        NM = NumbaModel2(self.sum_idx, self.sum_mapped_idx,
                        self.sum_mapped, self.compiled_eq_idxs,
                        self.index_helper, self.length, self.flat_scope_idx_from,
                        self.flat_scope_idx_from_idx_1,self.flat_scope_idx_from_idx_2,
                        self.flat_scope_idx, self.flat_scope_idx_idx_1,self.flat_scope_idx_idx_2,
                        self.scope_variables_flat, self.states_idx, self.derivatives_idx,
                        self.global_vars, self.scope_variables_2d)


        return NM.func


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
