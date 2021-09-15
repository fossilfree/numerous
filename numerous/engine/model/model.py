import ast
import itertools
from copy import copy

import numpy as np
import time
import inspect
import uuid

from numba.experimental import jitclass
import pandas as pd

from numerous.engine.model.utils import Imports, wrap_function, njit_and_compile_function
from numerous.engine.model.external_mappings import ExternalMapping, EmptyMapping

from numerous.utils.logger_levels import LoggerLevel

from numerous.utils.historian import InMemoryHistorian
from numerous.engine.model.graph_representation.mappings_graph import MappingsGraph
from numerous.engine.model.compiled_model import numba_model_spec, CompiledModel
from numerous.engine.system.connector import Connector

from numerous.engine.system.subsystem import Subsystem, ItemSet
from numerous.engine.variables import VariableType

from numerous.engine.model.ast_parser.parser_ast import parse_eq
from numerous.engine.model.graph_representation.graph import Graph
from numerous.engine.model.ast_parser.parser_ast import process_mappings

from numerous.engine.model.lowering.equations_generator import EquationGenerator
from numerous.engine.system import SetNamespace


class ModelNamespace:

    def __init__(self, tag, outgoing_mappings, item_tag, item_indcs, path, pos, item_path):
        self.tag = tag
        self.item_tag = item_tag
        self.outgoing_mappings = outgoing_mappings
        self.equation_dict = {}
        self.eq_variables_ids = []
        ##Ordered dictionary.
        self.variables = {}
        self.set_variables = None
        self.mappings = []
        self.full_tag = item_path + '.' + tag
        self.item_indcs = item_indcs

        self.path = path
        self.is_set = pos

    def ordered_variables(self):
        """
        return variables ordered for sequential llvm addressing
        """
        variables__ = []
        for vs in self.variable_scope:
            variables___ = []
            for v in vs:
                variables___.append(v)
            variables__.append(variables___)
        variables__ = [list(x) for x in zip(*variables__)]
        variables__ = list(itertools.chain(*variables__))
        return variables__


class ModelAssembler:

    @staticmethod
    def namespace_parser(input_namespace):
        equation_dict = {}
        tag, namespaces = input_namespace
        variables_ = {}
        for namespace in namespaces:

            for i, (eq_tag, eq_methods) in enumerate(namespace.equation_dict.items()):
                scope_id = "{0}_{1}_{2}".format(eq_tag, namespace.tag, tag, str(uuid.uuid4()))
                equation_dict.update({scope_id: (eq_methods, namespace.outgoing_mappings)})
            for v in namespace.ordered_variables():
                variables_.update({v.id: v})

        return variables_, equation_dict


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

    def __init__(self, system=None, logger_level=None, historian_filter=None, assemble=True, validate=False,
                 external_mappings=None, data_loader=None, imports=None, historian=InMemoryHistorian(),
                 use_llvm=True, save_to_file=False, generate_graph_pdf=False):

        self.path_to_variable = {}
        self.generate_graph_pdf = generate_graph_pdf

        if logger_level == None:
            self.logger_level = LoggerLevel.ALL
        else:
            self.logger_level = logger_level

        self.is_external_data = True if external_mappings else False
        self.external_mappings = ExternalMapping(external_mappings,
                                                 data_loader) if external_mappings else EmptyMapping()

        self.use_llvm = use_llvm
        self.save_to_file = save_to_file
        self.imports = Imports()
        self.imports.add_as_import("numpy", "np")
        self.imports.add_from_import("numba", "njit")
        self.imports.add_from_import("numba", "carray")
        self.imports.add_from_import("numba", "float64")
        self.imports.add_from_import("numba", "float32")
        if imports:
            for (k, v) in imports:
                self.imports.add_from_import(k, v)

        self.numba_callbacks_init = []
        self.numba_callbacks_variables = []
        self.numba_callbacks = []
        self.numba_callbacks_init_run = []
        self.callbacks = []
        self.historian_filter = historian_filter
        self.system = system
        self.event_function = None
        self.condition_function = None
        self.derivatives = {}
        self.model_items = {}
        self.state_history = {}
        self.compiled_eq = []
        self.flat_scope_idx = None
        self.flat_scope_idx_from = None
        self.historian_df = None
        self.aliases = {}
        self.historian = historian
        self.vars_ordered_value = {}

        self.global_variables_tags = ['time']
        self.global_vars = np.array([0], dtype=np.float64)

        self.equation_dict = {}
        self.variables = {}
        self.name_spaces = {}
        self.flat_variables = {}
        self.path_variables = {}
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

    def __get_mapping__variable(self, variable, depth):
        if variable.mapping and depth > 0:
            return self.__get_mapping__variable(variable.mapping, depth - 1)
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
        # scope_variables <variable_id, Variable>
        for variables, equation_dict in map(ModelAssembler.namespace_parser, model_namespaces):
            self.equation_dict.update(equation_dict)
            self.variables.update(variables)

        mappings = []
        for variable in self.variables.values():
            variable.top_item = self.system.id

        for scope_var_idx, var in enumerate(self.variables.values()):
            if var.mapping:
                _from = self.__get_mapping__variable(self.variables[var.mapping.id], depth=1)
                mappings.append((var.id, [_from.id]))
            if not var.mapping and var.sum_mapping:
                sum_mapping = []
                for mapping_id in var.sum_mapping:
                    _from = self.__get_mapping__variable(self.variables[mapping_id.id], depth=1)
                    sum_mapping.append(_from.id)
                mappings.append((var.id, sum_mapping))

        self.mappings_graph = Graph(preallocate_items=1000000)

        nodes_dep = {}
        self.equations_parsed = {}
        self.scoped_equations = {}
        self.equations_top = {}

        logging.info('parsing equations starting')
        for v in self.variables.values():
            v.top_item = self.system.id

        for ns in model_namespaces:
            ##will be false for empty namespaces. Ones without equations and variables.
            if ns[1]:
                ## Key : scope.tag Value: Variable or VariableSet
                if ns[1][0].is_set:
                    tag_vars = ns[1][0].set_variables
                else:
                    tag_vars = {v.tag: v for k, v in ns[1][0].variables.items()}

                parse_eq(model_namespace=ns[1][0], item_id=ns[0], mappings_graph=self.mappings_graph,
                         scope_variables=tag_vars, parsed_eq_branches=self.equations_parsed,
                         scoped_equations=self.scoped_equations, parsed_eq=self.equations_top)

        logging.info('parsing equations completed')

        # Process mappings add update the global graph
        self.mappings_graph = process_mappings(mappings, self.mappings_graph, self.variables)
        self.mappings_graph.build_node_edges()

        logging.info('Mappings processed')

        # Process variables
        states = []
        deriv = []
        mapping = []
        other = []

        for sv_id, sv in self.variables.items():
            if sv.logger_level is None:
                sv.logger_level = LoggerLevel.ALL
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

        logging.info('variables sorted')

        self.mappings_graph = MappingsGraph.from_graph(self.mappings_graph)
        self.mappings_graph.remove_chains()
        tmp_vars = self.mappings_graph.create_assignments(self.variables)
        self.mappings_graph.add_mappings()
        if self.generate_graph_pdf:
            self.mappings_graph.as_graphviz(self.system.tag, force=True)
        self.lower_model_codegen(tmp_vars)
        self.logged_aliases = {}

        for i, variable in enumerate(self.variables.values()):
            if variable.temporary_variable:
                continue
            if variable.logger_level is None:
                variable.logger_level = LoggerLevel.ALL
            logvar = False
            if variable.logger_level.value >= self.logger_level.value:
                logvar = True
            for path in variable.path.path[self.system.id]:
                self.aliases.update({path: variable.id})
                if logvar:
                    self.logged_aliases.update({path: variable.id})
            if variable.alias is not None:
                self.aliases.update({variable.alias: variable.id})
                if logvar:
                    self.logged_aliases.update({variable.alias: variable.id})
            for path in variable.path.path[self.system.id]:
                self.path_variables.update({path: variable.value})  # is this used at all?

        self.inverse_aliases = {v: k for k, v in self.aliases.items()}
        inverse_logged_aliases = {}  # {id: [alias1, alias2...], ...}
        for k, v in self.logged_aliases.items():
            inverse_logged_aliases[v] = inverse_logged_aliases.get(v, []) + [k]

        self.inverse_logged_aliases = inverse_logged_aliases
        self.logged_variables = {}

        for varname, ix in self.vars_ordered_values.items():  # now it's a dict...
            var = self.variables[varname]
            if var.logger_level.value >= self.logger_level.value:
                if varname in self.inverse_logged_aliases:
                    for vv in self.inverse_logged_aliases[varname]:
                        self.logged_variables.update({vv: ix})

        number_of_external_mappings = 0
        external_idx = []

        for var in self.variables.values():
            if self.external_mappings.is_mapped_var(self.variables, var.id, self.system.id):
                external_idx.append(self.vars_ordered_values[var.id])
                number_of_external_mappings += 1
                self.external_mappings.add_df_idx(self.variables, var.id, self.system.id)

        self.number_of_external_mappings = number_of_external_mappings
        self.external_mappings.store_mappings()
        self.external_idx = np.array(external_idx, dtype=np.int64)
        self.generate_path_to_varaible()

        assemble_finish = time.time()
        print("Assemble time: ", assemble_finish - assemble_start)
        self.info.update({"Assemble time": assemble_finish - assemble_start})
        self.info.update({"Number of items": len(self.model_items)})
        self.info.update({"Number of variables": len(self.variables)})
        self.info.update({"Number of equation scopes": len(self.equation_dict)})
        self.info.update({"Number of equations": len(self.compiled_eq)})
        self.info.update({"Solver": {}})

    def lower_model_codegen(self, tmp_vars):

        logging.info('lowering model')

        eq_gen = EquationGenerator(equations=self.equations_parsed, filename="kernel.py",
                                   equation_graph=self.mappings_graph,
                                   scope_variables=self.variables, scoped_equations=self.scoped_equations,
                                   temporary_variables=tmp_vars, system_tag=self.system.tag, use_llvm=self.use_llvm,
                                   imports=self.imports)

        compiled_compute, var_func, var_write, self.vars_ordered_values, self.variables, \
        self.state_idx, self.derivatives_idx = \
            eq_gen.generate_equations(save_to_file=self.save_to_file)

        for varname, ix in self.vars_ordered_values.items():
            var = self.variables[varname]
            var.llvm_idx = ix
            var.model = self
            if getattr(var, 'logger_level',
                       None) is None:  # added to temporary variables - maybe put in generate_equations?
                setattr(var, 'logger_level', LoggerLevel.ALL)

        def c1(self, array_):
            return compiled_compute(array_)

        def c2(self):
            return var_func()

        def c3(self, value, idx):
            return var_write(value, idx)

        setattr(CompiledModel, "compiled_compute", c1)
        setattr(CompiledModel, "read_variables", c2)
        setattr(CompiledModel, "write_variables", c3)

        self.compiled_compute, self.var_func, self.var_write = compiled_compute, var_func, var_write
        self.init_values = np.ascontiguousarray(
            [self.variables[k].value for k in self.vars_ordered_values.keys()],
            dtype=np.float64)
        self.update_all_variables()

        # values of all model variables in specific order: self.vars_ordered_values
        # full tags of all variables in the model in specific order: self.vars_ordered
        # dict with scope variable id as key and scope variable itself as value

        # Create aliases for all paths in each scope variable
        def c4(values_dict):
            return [self.var_write(v, self.vars_ordered_values[self.aliases[k]]) for k, v in values_dict.items()]

        def c5():
            vals = var_func()
            return {self.inverse_aliases[k]: v for k, v in zip(self.vars_ordered_values.keys(), vals)}

        setattr(self, "update_variables", c4)
        setattr(self, "get_variables", c5)

        self.info.update({"Solver": {}})

    def generate_path_to_varaible(self):
        for k, v in self.aliases.items():
            self.path_to_variable[k] = self.variables[v]

    def update_all_variables(self):
        for k, v in self.vars_ordered_values.items():
            self.var_write(self.variables[k].value, v)

    def update_local_variables(self):
        vars = self.numba_model.read_variables()
        for k, v in self.vars_ordered_values.items():
            self.variables[k].value = vars[v]

    def get_states(self):
        """

        Returns
        -------
        states : list of states
            list of all states.
        """
        return self.variables[self.state_idx]

    def update_states(self, y):
        self.variables[self.states_idx] = y

    def history_as_dataframe(self):
        time = self.data[0]
        data = {'time': time}

        for i, var in enumerate(self.var_list):
            data.update({var: self.data[i + 1]})

        self.df = pd.DataFrame(data)
        self.df = self.df.dropna(subset=['time'])
        self.df = self.df.set_index('time')
        self.df.index = pd.to_timedelta(self.df.index, unit='s')

    def validate_bindings(self):
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

    @property
    def states_as_vector(self):
        """
        Returns current states values.

        Returns
        -------
        state_values : array of state values

        """
        return self.var_func()[self.state_idx]

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

    def add_event(self, key, condition, action, terminal=True, direction=-1):
        condition = self._replace_path_strings(condition, "state")

        condition.terminal = terminal
        condition.direction = direction
        action = self._replace_path_strings(action, "var")
        # njit_and_compile_function(func, self.imports.from_imports)
        self.events.append((key, condition, action))


    # def compile_events(self):
    #     @njit
    #     def condition(t, states):
    #         result = []
    #         result.append(f_name(t, states))
    #         result.append(f_name(t, states))
    #         result.append(f_name(t, states))
    #         result.append(f_name(t, states))
    #         return np.array(result, np.float64)
    #
    #     @njit
    #     def action(t, variables, a_idx):
    #         for idx, (_, _, ac) in enumerate(events):
    #             if a_idx == idx: ac(t, variables)

    def _get_var_idx(self, var, idx_type):
        if idx_type == "state":
            return np.where(self.state_idx == var.llvm_idx)[0]
        if idx_type == "var":
            return [var.llvm_idx]

    def _replace_path_strings(self, function, idx_type):
        lines = inspect.getsource(function)
        for (var_path, var) in self.path_to_variable.items():
            if var_path in lines:
                lines = lines.replace('[\'' + var_path + '\']', str(self._get_var_idx(var, idx_type)))
        func = ast.parse(lines.strip()).body[0]
        return func

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

    def create_model_namespaces(self, item):
        namespaces_list = []
        for namespace in item.registered_namespaces.values():
            set_namespace = isinstance(namespace, SetNamespace)
            model_namespace = ModelNamespace(namespace.tag, namespace.outgoing_mappings, item.tag, namespace.items,
                                             namespace.path, set_namespace, '.'.join(item.path))
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
            model_namespace.variables = {v.id: v for vs in model_namespace.variable_scope for v in vs}
            namespaces_list.append(model_namespace)
        return namespaces_list

    # Method that generates numba_model
    def generate_compiled_model(self, start_time, number_of_timesteps):
        for spec_dict in self.numba_callbacks_variables:
            for item in spec_dict.items():
                numba_model_spec.append(item)

        # Creating a copy of CompiledModel class so it is possible
        # to creat instance detached from muttable type of CompiledModel
        tmp = type(f'{CompiledModel.__name__}' + self.system.id, CompiledModel.__bases__, dict(CompiledModel.__dict__))
        if self.use_llvm:
            @jitclass(numba_model_spec)
            class CompiledModel_instance(tmp):
                pass
        else:
            class CompiledModel_instance(tmp):
                pass

        NM_instance = CompiledModel_instance(self.init_values, self.derivatives_idx, self.state_idx,
                                             self.global_vars, number_of_timesteps, start_time,
                                             self.historian.get_historian_max_size(number_of_timesteps),
                                             self.historian.need_to_correct(),
                                             self.external_mappings.external_mappings_time,
                                             self.number_of_external_mappings,
                                             self.external_mappings.external_mappings_numpy,
                                             self.external_mappings.external_df_idx,
                                             self.external_mappings.interpolation_info,
                                             self.is_external_data, self.external_mappings.t_max,
                                             self.external_idx
                                             )

        for key, value in self.path_variables.items():
            NM_instance.path_variables[key] = value
            NM_instance.path_keys.append(key)
        # NM_instance.run_init_callbacks(start_time)
        NM_instance.map_external_data(start_time)

        # NM_instance.historian_update(start_time)
        self.numba_model = NM_instance
        return self.numba_model

    def create_historian_dict(self, historian_data=None):
        if historian_data is None:
            historian_data = self.numba_model.historian_data
        time = historian_data[0]

        data = {var: historian_data[i + 1] for var, i in self.logged_variables.items()}
        data.update({'time': time})
        return data

    def create_historian_df(self):
        self.historian_df = self._generate_history_df(self.numba_model.historian_data, rename_columns=False)
        self.historian.store(self.historian_df)

    def _generate_history_df(self, historian_data, rename_columns=True):
        data = self.create_historian_dict(historian_data)
        return AliasedDataFrame(data, aliases=self.aliases, rename_columns=True)


class AliasedDataFrame(pd.DataFrame):
    _metadata = ['aliases']

    def __init__(self, data, aliases={}, rename_columns=True):
        super().__init__(data)
        self.aliases = aliases
        self.rename_columns = rename_columns
        if self.rename_columns:
            tmp = copy(list(data.keys()))
            for key in tmp:
                if key in self.aliases.keys():
                    data[self.aliases[key]] = data.pop(key)
            super().__init__(data)

    def __getitem__(self, item):
        if self.rename_columns:
            if not isinstance(item, list):
                col = self.aliases[item] if item in self.aliases else item
                return super().__getitem__(col)

            cols = [self.aliases[i] if i in self.aliases else i for i in item]

            return super().__getitem__(cols)
        else:
            return super().__getitem__(item)
