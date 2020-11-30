import itertools
import numpy as np
import time
import uuid

from numba.experimental import jitclass
import pandas as pd

from model.graph_representation.equation_graph import EquationGraph
from numerous.engine.model.compiled_model import numba_model_spec, CompiledModel
from numerous.engine.system.connector import Connector
from numerous.engine.scope import Scope, ScopeVariable, ScopeSet

from numerous.engine.system.subsystem import Subsystem, ItemSet
from numerous.engine.variables import VariableType
from numerous.utils.numba_callback import NumbaCallbackBase

import operator

from enum import IntEnum
from model.graph_representation.parser_ast import parse_eq
from model.graph_representation.graph import Graph
from model.graph_representation.parser_ast import process_mappings

from model.lowering.equations_generator import EquationGenerator
from numerous.engine.system import SetNamespace


class LowerMethod(IntEnum):
    Tensor = 0
    Codegen = 1


lower_method = LowerMethod.Codegen


class ModelNamespace:

    def __init__(self, tag, outgoing_mappings, item_tag, item_indcs, path, pos):
        self.tag = tag
        self.item_tag = item_tag
        self.outgoing_mappings = outgoing_mappings
        self.equation_dict = {}
        self.eq_variables_ids = []
        self.variables = {}
        self.set_variables = None
        self.mappings = []
        self.full_tag = item_tag + '.' + tag
        self.item_indcs = item_indcs

        self.path = path
        self.is_set = pos


class ModelAssembler:

    @staticmethod
    def __create_scope(eq_tag, eq_methods, eq_variables, namespace, tag, variables):
        scope_id = "{0}_{1}_{2}".format(eq_tag, namespace.tag, tag, str(uuid.uuid4()))
        if namespace.is_set:
            scope = ScopeSet(scope_id, namespace.item_indcs)
        else:
            scope = Scope(scope_id, namespace.item_indcs)
        for variable in eq_variables:
            scope.add_variable(variable)
            variable.bound_equation_methods = eq_methods
            if variable.mapping:
                pass
            variable.parent_scope_id = scope_id
            variables.update({variable.id: variable})
        return scope

    @staticmethod
    def namespace_parser(input_namespace):
        scope_select = {}
        variables = {}
        equation_dict = {}
        tag, namespaces = input_namespace
        variables_ = {}
        for namespace in namespaces:
            for i, (eq_tag, eq_methods) in enumerate(namespace.equation_dict.items()):
                scope = ModelAssembler.__create_scope(eq_tag, eq_methods,
                                                      [v for i_ in namespace.variable_scope for v in i_],
                                                      namespace, tag, variables)
                scope_select.update({scope.id: scope})
                equation_dict.update({scope.id: (eq_methods, namespace.outgoing_mappings)})
            variables_.update({v.id: v for vs in namespace.variable_scope for v in vs})

        return variables_, scope_select, equation_dict


import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class Model:
    """
     The model object traverses the system to collect all information needed to pass to the solver
     for computation – the model also back-propagates the numerical results from the solver into the system,
     so they can be accessed as variable values there.
    """

    def __init__(self, system=None, historian_filter=None, assemble=True, validate=False):

        self.numba_callbacks_init = []
        self.numba_callbacks_variables = []
        self.numba_callbacks = []
        self.numba_callbacks_init_run = []
        self.callbacks = []
        self.historian_filter = historian_filter
        self.system = system
        self.events = {}
        self.derivatives = {}
        self.model_items = {}
        self.state_history = {}
        self.synchronized_scope = {}
        self.compiled_eq = []
        self.flat_scope_idx = None
        self.flat_scope_idx_from = None
        self.historian_df = None

        self.global_variables_tags = ['time']
        self.global_vars = np.array([0], dtype=np.float64)

        # LNT: need to map each var id to set variable
        self.variables_set_var = {}

        self.equation_dict = {}
        self.scope_variables = {}
        self.name_spaces = {}
        self.variables = {}
        self.flat_variables = {}
        self.path_variables = {}
        self.path_scope_variables = {}
        self.states = {}
        self.period = 1
        self.mapping_from = []
        self.mapping_to = []
        self.eq_outgoing_mappings = []
        self.sum_mapping_from = []
        self.sum_mapping_to = []
        self.states_idx = []
        self.derivatives_idx = []
        self.scope_to_variables_idx = []
        self.numba_model = None

        self.info = {}
        if assemble:
            self.assemble()

        if validate:
            self.validate()

    def __add_item(self, item):
        model_namespaces = []
        if item.id in self.model_items:
            return model_namespaces

        if item.callbacks:
            self.callbacks.append(item.callbacks)

        self.model_items.update({item.id: item})
        model_namespaces.append((item.id, self.create_model_namespaces(item)))
        if isinstance(item, ItemSet):
            return model_namespaces
        if isinstance(item, Connector):
            for binded_item in item.get_binded_items():
                if not binded_item.part_of_set:
                    model_namespaces.extend(self.__add_item(binded_item))
        if isinstance(item, Subsystem):
            for registered_item in item.registered_items.values():
                model_namespaces.extend(self.__add_item(registered_item))
        return model_namespaces

    def __get_mapping__variable(self, variable):
        if variable.mapping:
            return self.__get_mapping__variable(variable.mapping)
        else:
            return variable

    def assemble(self):
        """
        Assembles the model.
        """
        """
        notation:
        - _idx for single integers / tuples, 
        - _idxs for lists / arrays of integers
        - _pos as counterpart to _from
        -  _flat
        -  _3d 

        """

        def __get_mapping__idx(variable):
            if variable.mapping:
                return __get_mapping__idx(variable.mapping)
            else:
                return variable.idx_in_scope[0]

        logging.info("Assembling numerous Model")
        assemble_start = time.time()

        # 1. Create list of model namespaces
        model_namespaces = [_ns
                            for item in self.system.registered_items.values() if not item.part_of_set
                            for _ns in self.__add_item(item)]

        # 2. Compute dictionaries
        # equation_dict <scope_id, [Callable]>
        # synchronized_scope <scope_id, Scope>
        # scope_variables <variable_id, Variable>
        for variables, scope_select, equation_dict in map(ModelAssembler.namespace_parser, model_namespaces):
            self.equation_dict.update(equation_dict)
            self.synchronized_scope.update(scope_select)
            self.scope_variables.update(variables)

        mappings = []

        def __get_mapping__variable(variable):
            if variable.mapping:
                return __get_mapping__variable(variable.mapping)
            else:
                return variable

        for scope_var_idx, var in enumerate(self.scope_variables.values()):
            if var.mapping:
                _from = self.__get_mapping__variable(self.variables[var.mapping.id])
                mappings.append((var.id, [_from.id]))
            if not var.mapping and var.sum_mapping:
                sum_mapping = []
                for mapping_id in var.sum_mapping:
                    _from = self.__get_mapping__variable(self.variables[mapping_id.id])
                    sum_mapping.append(_from.id)
                mappings.append((var.id, sum_mapping))

        self.eg = Graph(preallocate_items=1000000)

        nodes_dep = {}
        self.equations_parsed = {}
        self.scoped_equations = {}
        self.equations_top = {}

        logging.info('parsing equations starting')
        for v in self.scope_variables.values():
            v.top_item = self.system.id

        for ns in model_namespaces:
            tag_vars = {v.tag: v for v in self.scope_variables.values()}
            tag_vars_ = {v.tag: v for k, v in ns[1][0].variables.items()}
            parse_eq(ns[1][0],self.eg, nodes_dep, tag_vars, self.equations_parsed, self.scoped_equations,
                     self.equations_top, tag_vars_)

        logging.info('parsing equations completed')

        # Process mappings add update the global graph
        self.eg = process_mappings(mappings,  self.eg, nodes_dep, self.scope_variables)
        self.eg.build_node_edges()

        logging.info('Mappings processed')

        # Process variables
        states = []
        deriv = []
        mapping = []
        other = []

        for sv_id, sv in self.scope_variables.items():
            if sv.type == VariableType.STATE:
                states.append(sv)
            elif sv.type == VariableType.DERIVATIVE:
                deriv.append(sv)
            elif sv.sum_mapping or sv.mapping:
                mapping.append(sv)
            else:
                other.append(sv)

        self.vars_ordered = states + deriv + mapping + other
        self.states_end_ix = len(states)

        self.deriv_end_ix = self.states_end_ix + len(deriv)
        self.mapping_end_ix = self.deriv_end_ix + len(mapping)

        self.special_indcs = [self.states_end_ix, self.deriv_end_ix, self.mapping_end_ix]
        self.vars_ordered_values = np.array([v.value for v in self.vars_ordered], dtype=np.float64)

        if self.gg:
            vars_node_id = {self.gg.get(v, 'scope_var').id: k for k, v in self.gg.node_map.items() if
                            self.gg.get(v, 'scope_var')}

            self.vars_ordered_map = []

            for v in self.vars_ordered:
                if v.id in vars_node_id:
                    self.vars_ordered_map.append(vars_node_id[v.id])
                else:
                    self.vars_ordered_map.append(v.id.replace('-', '_'))

        logging.info('variables sorted')

        self.eg = EquationGraph.from_graph(self.eg)
        self.eg.remove_chains()
        self.eg.create_assignments(self.scope_variables)
        self.eg.add_mappings()

        if lower_method == LowerMethod.Codegen:
            self.lower_model_codegen()
            self.generate_numba_model = self.generate_numba_model_code_gen
        elif lower_method == LowerMethod.Tensor:
            self.lower_model_tensor()
            self.generate_numba_model = self.generate_numba_model_tensor

        assemble_finish = time.time()
        print("Assemble time: ", assemble_finish - assemble_start)
        self.info.update({"Assemble time": assemble_finish - assemble_start})
        self.info.update({"Number of items": len(self.model_items)})
        self.info.update({"Number of variables": len(self.scope_variables)})
        self.info.update({"Number of equation scopes": len(self.equation_dict)})
        self.info.update({"Number of equations": len(self.compiled_eq)})
        self.info.update({"Solver": {}})

    def lower_model_codegen(self):

        logging.info('lowering model')
        eq_gen = EquationGenerator(equations=self.equations_parsed, filename="kernel.py", equation_graph=self.eg,
                                   scope_variables=self.scope_variables,scoped_equations=self.scoped_equations)

        self.compiled_compute, self.var_func, self.vars_ordered_values, self.vars_ordered, self.scope_vars_vars = \
            eq_gen.generate_equations()

        # values of all model variables in specific order: self.vars_ordered_values
        # full tags of all variables in the model in specific order: self.vars_ordered
        # dict with scope variable id as key and scope variable itself as value

        # Create aliases for all paths in each scope variable
        self.aliases = {}
        self.path_variables_aliases = {}
        for sv in self.scope_variables.values():

            if self.system.id in sv.path.path:
                aliases__ = [a for p in sv.path.path.values() for a in p]

                self.path_variables_aliases[sv.get_path_dot()] = aliases__

        for k, v in self.path_variables_aliases.items():
            for a in v:
                self.aliases[a] = k

        self.info.update({"Solver": {}})

    def lower_model_tensor(self):
        pass


    def get_states(self):
        """

        Returns
        -------
        states : list of states
            list of all states.
        """
        return self.scope_variables[self.states_idx]

    def synchornize_variables(self):
        '''
        Updates all the values of all Variable instances stored in
        `self.variables` with the values stored in `self.scope_vars_3d`.
        '''
        for variable, value in zip(self.variables.values(),
                                   self.scope_vars_3d[self.var_idxs_pos_3d]):
            variable.value = value

    def update_states(self, y):
        self.scope_variables[self.states_idx] = y

    def history_as_dataframe(self):
        time = self.data[0]
        data = {'time': time}

        for i, var in enumerate(self.var_list):
            data.update({var: self.data[i + 1]})

        self.df = pd.DataFrame(data)
        self.df = self.df.dropna(subset=['time'])
        self.df = self.df.set_index('time')
        self.df.index = pd.to_timedelta(self.df.index, unit='s')

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
        self.scope_vars_3d[self.state_idxs_3d] = r1

    @property
    def states_as_vector(self):
        """
        Returns current states values.

        Returns
        -------
        state_values : array of state values

        """
        # return self.scope_vars_3d[self.state_idxs_3d]
        return self.vars_ordered_values[0:self.states_end_ix]

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

    def add_callback(self, callback_class: NumbaCallbackBase) -> None:
        """

        """

        self.callbacks.append(callback_class)
        numba_update_function = Equation_Parser.parse_non_numba_function(callback_class.update, r"@NumbaCallback.+")
        self.numba_callbacks.append(numba_update_function)

        if callback_class.update.run_after_init:
            self.numba_callbacks_init_run.append(numba_update_function)

        numba_initialize_function = Equation_Parser.parse_non_numba_function(callback_class.initialize,
                                                                             r"@NumbaCallback.+")

        self.numba_callbacks_init.append(numba_initialize_function)
        self.numba_callbacks_variables.append(callback_class.numba_params_spec)

    def create_model_namespaces(self, item):
        namespaces_list = []
        for namespace in item.registered_namespaces.values():
            set_namespace = isinstance(namespace,SetNamespace)
            model_namespace = ModelNamespace(namespace.tag, namespace.outgoing_mappings, item.tag, namespace.items,
                                             namespace.path, set_namespace)
            # model_namespace.mappings = namespace.mappings
            model_namespace.variable_scope = namespace.get_flat_variables()
            model_namespace.set_variables = namespace.set_variables

            equation_dict = {}
            eq_variables_ids = []
            for eq in namespace.associated_equations.values():
                equations = []
                ids = []
                for equation in eq.equations:
                    equations.append(equation)
                for vardesc in eq.variables_descriptions:
                    if set_namespace:
                        for item in namespace.items:
                            variable = item.registered_namespaces[namespace.tag].get_variable(vardesc.tag)
                            ids.append(variable.id)
                    else:
                        variable = namespace.get_variable(vardesc.tag)
                        ids.append(variable.id)
                equation_dict.update({eq.tag: equations})
                eq_variables_ids.append(ids)
            model_namespace.equation_dict = equation_dict
            model_namespace.variables = {v.id: ScopeVariable(v) for vs in model_namespace.variable_scope for v in vs}
            self.variables.update(model_namespace.variables)
            namespaces_list.append(model_namespace)
        return namespaces_list

    # Method that generates numba_model
    def generate_numba_model_code_gen(self, start_time, number_of_timesteps):
        from numba import float64, int64
        compute = self.compiled_compute
        var_func = self.var_func
        spec = [
            ('variables', float64[:]),
            ('historian_data', float64[:, :]),
            ('historian_ix', int64),

        ]

        @jitclass(spec)
        class CompiledModel:
            def __init__(self, variables_init):
                self.variables = variables_init
                self.historian_data = np.zeros((number_of_timesteps, len(self.variables) + 1), dtype=np.float64)
                # self.historian_data = np.zeros((2,2), dtype=np.float64)
                self.historian_ix = 0

            def func(self, _t, y):
                # print('diff: ', _t)
                d = compute(y)
                # for d_ in d:
                #     print(d_)
                # raise ValueError('')
                return d

            def historian_update(self, t):
                # print('hist update')

                self.variables[:] = var_func(np.int64(0))

                self.historian_data[self.historian_ix][0] = t
                self.historian_data[self.historian_ix][1:] = self.variables[:]

                self.historian_ix += 1

        self.numba_model = CompiledModel(self.vars_ordered_values)

        self.numba_model.func(0, self.vars_ordered_values[0:self.states_end_ix])

        logging.info('completed numba model')

        return self.numba_model

    def generate_numba_model_tensor(self, start_time, number_of_timesteps):

        for spec_dict in self.numba_callbacks_variables:
            for item in spec_dict.items():
                numba_model_spec.append(item)

        def create_eq_call(eq_method_name: str, i: np.int64):
            return "      self." \
                   "" + eq_method_name + "(array_3d[" + str(i) + \
                   ", :self.num_uses_per_eq[" + str(i) + "]])\n"

        Equation_Parser.create_numba_iterations(NumbaModel, self.compiled_eq, "compute_eq", "func"
                                                , create_eq_call, "array_3d", map_sorting=self.eq_outgoing_mappings)

        ##Adding callbacks_varaibles to numba specs
        def create_cbi_call(_method_name: str, i: np.int64):
            return "      self." \
                   "" + _method_name + "(time, self.path_variables)\n"

        Equation_Parser.create_numba_iterations(NumbaModel, self.numba_callbacks, "run_callbacks", "callback_func"
                                                , create_cbi_call, "time")

        def create_cbi2_call(_method_name: str, i: np.int64):
            return "      self." \
                   "" + _method_name + "(self.number_of_variables,self.number_of_timesteps)\n"

        Equation_Parser.create_numba_iterations(NumbaModel, self.numba_callbacks_init, "init_callbacks",
                                                "callback_func_init_", create_cbi2_call, "")

        def create_cbiu_call(_method_name: str, i: np.int64):
            return "      self." \
                   "" + _method_name + "(time, self.path_variables)\n"

        Equation_Parser.create_numba_iterations(NumbaModel, self.numba_callbacks_init_run, "run_init_callbacks",
                                                "callback_func_init_pre_update", create_cbiu_call, "time")

        @jitclass(numba_model_spec)
        class NumbaModel_instance(NumbaModel):
            pass

        NM_instance = NumbaModel_instance(self.var_idxs_pos_3d, self.var_idxs_pos_3d_helper,
                                          np.int64(len(self.compiled_eq)), self.state_idxs_3d[0].shape[0],
                                          self.differing_idxs_pos_3d[0].shape[0], self.scope_vars_3d,
                                          self.state_idxs_3d,
                                          self.deriv_idxs_3d, self.differing_idxs_pos_3d, self.differing_idxs_from_3d,
                                          np.int64(self.num_uses_per_eq), self.sum_idxs_pos_3d, self.sum_idxs_sum_3d,
                                          self.sum_slice_idxs, self.sum_mapped_idxs_len, self.sum_mapping,
                                          self.global_vars, number_of_timesteps, len(self.path_variables), start_time,
                                          self.mapped_variables_array)

        for key, value in self.path_variables.items():
            NM_instance.path_variables[key] = value
            NM_instance.path_keys.append(key)
        NM_instance.run_init_callbacks(start_time)

        NM_instance.historian_update(start_time)
        self.numba_model = NM_instance

        return self.numba_model

    def create_historian_df(self):
        # _time = self.numba_model.historian_data[0]
        # data = {'time': _time}
        #
        # for i, var in enumerate(self.path_variables):
        #     data.update({var: self.numba_model.historian_data[i + 1]})
        #
        # self.historian_df = pd.DataFrame(data)
        # self.historian_df = self.historian_df.dropna(subset=['time'])
        # self.historian_df = self.historian_df.set_index('time')
        # self.historian_df.index = pd.to_timedelta(self.historian_df.index, unit='s')

        if lower_method == LowerMethod.Codegen:
            time = self.numba_model.historian_data[:, 0]
            data = {'time': time}

            for i, var in enumerate(self.vars_ordered):
                # data.update({".".join(self.variables[var.id].path.path[self.system.id]): self.numba_model.historian_data[:,i+1]})
                # print(var)
                data.update({
                    var: self.numba_model.historian_data[:, i + 1]})

            self.historian_df = AliasedDataFrame(data, aliases=self.aliases)

        if lower_method == LowerMethod.Tensor:
            time = self.numba_model.historian_data[0]
            data = {'time': time}

            for i, var in enumerate(self.path_variables):
                data.update({var: self.numba_model.historian_data[i + 1]})

            self.historian_df = pd.DataFrame(data)
        # self.df.set_index('time')


class AliasedDataFrame(pd.DataFrame):
    _metadata = ['aliases']

    def __init__(self, data, aliases={}):
        # df['system.alias.T']

        self.aliases = aliases
        # print(data.keys())

        super().__init__(data)

    def __getitem__(self, item):
        if not isinstance(item, list):
            col = self.aliases[item] if item in self.aliases else item
            return super().__getitem__(col)

        cols = [self.aliases[i] if i in self.aliases else i for i in item]
        # print('cols: ',cols)

        return super().__getitem__(cols)
