import ast
import itertools
import numpy as np
import operator
import re
import time
import uuid

from numba import jitclass
import pandas as pd
from numerous.engine.model.equation_parser import Equation_Parser
from numerous.engine.model.numba_model import numba_model_spec, NumbaModel
from numerous.engine.system.connector import Connector
from examples.historyDataFrameCallbackExample import HistoryDataFrameCallback
from numerous.engine.scope import Scope, ScopeVariable
# from numerous.engine.simulation.simulation_callbacks import _SimulationCallback, _Event

from numerous.engine.system.subsystem import Subsystem
from numerous.engine.variables import VariableType
from numerous.utils.numba_callback import NumbaCallbackBase

import operator

from enum import IntEnum, unique
from numerous.engine.model.parser_ast import parse_eq
#from numerous_graph.graph import Graph
from numerous.engine.model.graph import Graph
from numerous.engine.model.parser_ast import process_mappings
from numerous.engine.model.generate_model import generate

from numerous.engine.model.generate_equations import generate_equations

class LowerMethod(IntEnum):
    Tensor=0
    Codegen=1


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
        #self.full_tag = item_tag + '_' + tag
        self.full_tag = item_tag + '.' + tag
        self.item_indcs = item_indcs

        self.path = path
        self.part_of_set = pos

    def get_path_dot(self):
        return ".".join(self.path)


class ModelAssembler:

    @staticmethod
    def __create_scope(eq_tag, eq_methods, eq_variables, namespace, tag, variables):
        scope_id = "{0}_{1}_{2}".format(eq_tag, namespace.tag, tag, str(uuid.uuid4()))
        scope = Scope(scope_id, namespace.item_indcs)
        for variable in eq_variables:
            scope.add_variable(variable)
            variable.bound_equation_methods = eq_methods
            if variable.mapping:
                pass
            variable.parent_scope_id = scope_id
            # Needed for updating states after solve run
            #if variable.type.value == VariableType.STATE.value:
            #    variable.associated_state_scope.append(scope_id)
            variables.update({variable.id: variable})
        return scope

    @staticmethod
    def t_1(input_namespace):
        scope_select = {}
        variables = {}
        equation_dict = {}
        name_spaces_dict = {}
        tag, namespaces = input_namespace
        variables_ = {}
        for namespace in namespaces:
            for i, (eq_tag, eq_methods) in enumerate(namespace.equation_dict.items()):
                scope = ModelAssembler.__create_scope(eq_tag, eq_methods,
                                                       [v for i_ in namespace.variable_scope for v in i_],
                                                      namespace, tag, variables)
                scope_select.update({scope.id: scope})
                equation_dict.update({scope.id: (eq_methods, namespace.outgoing_mappings)})
                name_spaces_dict.update({scope.id: input_namespace})
            variables_.update({v.id: v for vs in namespace.variable_scope for v in vs})

        return variables_, scope_select, equation_dict, name_spaces_dict


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

        #LNT: need to map each var id to set variable
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
        if isinstance(item, Connector):
            for binded_item in item.get_binded_items():
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

        for item in self.system.registered_items.values():
            for ns in item.registered_namespaces.values():
                print('ns: ',ns.get_path_dot())
                #if not ns.part_of_set:
                ns.update_set_var()

        self.system.update_variables_path_()
        # 1. Create list of model namespaces
        model_namespaces = [_ns
                            for item in self.system.registered_items.values()
                            for _ns in self.__add_item(item)]

        # 2. Compute dictionaries
        # equation_dict <scope_id, [Callable]>
        # synchronized_scope <scope_id, Scope>
        # scope_variables <variable_id, Variable>
        for variables, scope_select, equation_dict, name_space in map(ModelAssembler.t_1, model_namespaces):
            self.equation_dict.update(equation_dict)
            self.synchronized_scope.update(scope_select)
            self.scope_variables.update(variables)
            self.name_spaces.update(name_space)

        for sv in self.scope_variables.values():
            print(sv.set_var,'-', sv.set_var_ix)


        self.mappings = []
        def __get_mapping__variable(variable):
            if variable.mapping:
                return __get_mapping__variable(variable.mapping)
            else:
                return variable

        for scope_var_idx, var in enumerate(self.scope_variables.values()):
            if var.mapping:
                _from = self.__get_mapping__variable(self.variables[var.mapping.id])
                self.mappings.append((var.id, [_from.id]))
            if not var.mapping and var.sum_mapping:
                sum_mapping = []
                for mapping_id in var.sum_mapping:

                    _from = self.__get_mapping__variable(self.variables[mapping_id.id])
                    sum_mapping.append(_from.id)
                self.mappings.append((var.id, sum_mapping))



        #print(self.equation_dict)
        self.gg = None#Graph(preallocate_items=1000000)
        self.eg = Graph(preallocate_items=1000000)



        self.scope_ids = {}
        for s in self.synchronized_scope.keys():
            #print(type(self.name_spaces[s][1][0]))
            s_id = f'{self.name_spaces[s][1][0].full_tag}'
            #print('k: ', s_id)
            s_id_ = s_id
            count = 1
            while s_id_ in list(self.scope_ids.values()):
                s_id_ = s_id + '_' + str(count)
                count += 1

            self.scope_ids[s] = s_id_
            #print(s_id_)
        #sdfsd=sdfsdf
        scope_item_tag = {}

        nodes_dep = {}
        self.equations_parsed = {}
        self.scoped_equations = {}
        self.equations_top = {}
        logging.info('parsing equations starting')
        for v in self.scope_variables.values():
            v.top_item = self.system.id

        for ns in self.name_spaces.values():
            try:
                tag_vars = {v.tag: v for v in self.scope_variables.values() if v.set_var in ns[1][0].set_variables}
            except:
                debug=1

            #print('scope_id: ', scope_id)
            #s_id = f's{self.scope_ids.index(scope_id)}'
            #print(eq)
            #if len(eq[0])>0:

            parse_eq(ns[1][0], self.gg, self.eg, nodes_dep, tag_vars, self.equations_parsed, self.scoped_equations, self.equations_top)

        logging.info('parsing equations completed')
        #for n in gg.nodes:
        #    print(n[0])

        #print('mapping: ',self.mappings)
        #self.eg.as_graphviz('eg', force=True)
        #Process mappings add update the global graph
        self.aliases, self.eg = process_mappings(self.mappings, self.gg, self.eg, nodes_dep, self.scope_variables, self.scope_ids)
        #self.eg.as_graphviz('eg_m', force=True)
        self.eg.build_node_edges()
        #self.eg.as_graphviz('equation_graph')
        logging.info('Mappings processed')
        #equation_graph_simplified.as_graphviz('equation_graph_simplified')
        #for n in equation_graph_simplified.get_nodes():
        #    print(n)
        #equation_graph_simplified.topological_nodes()
        logging.info('Checked topo sort of simple graph')
        #self.gg.topological_nodes()
        logging.info('Sorted gloabbal topo')
        #self.eg.as_graphviz('eq', force=True)
        #logging.info('rendered equation graph')

        #self.gg.as_graphviz('global_graph', force=True)
        #logging.info('rendered global graph')


        #nodes = self.gg.get_nodes()
        #for n in nodes:
        #    print(n[0], ' ', n[1].scope_var.type if hasattr(n[1], 'scope_var') and n[1].scope_var else "No type?!")



    #Process variables
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

        self.deriv_end_ix = self.states_end_ix+len(deriv)
        self.mapping_end_ix = self.deriv_end_ix + len(mapping)

        self.special_indcs = [self.states_end_ix, self.deriv_end_ix, self.mapping_end_ix]
        #print('gg nodes: ', gg.nodes)
        self.vars_ordered_values = np.array([v.value for v in self.vars_ordered], dtype=np.float64)

        #vars_node_id = {n[2]: n[0] for n in self.gg.get_nodes() if n[2]}
        if self.gg:
            vars_node_id = {self.gg.get(v, 'scope_var').id: k for k, v in self.gg.node_map.items() if self.gg.get(v, 'scope_var')}


            count = 0

            self.vars_ordered_map = []
            for v in self.vars_ordered:
                if v.id in vars_node_id:
                    self.vars_ordered_map.append(vars_node_id[v.id])
                else:
                    #vars_ordered_map.append(f'dummy__{count}')
                    self.vars_ordered_map.append(v.id.replace('-','_'))#.replace('.','_'))
                    count+=1

        logging.info('variables sorted')
        # #dsdfsdfsd = sdfsdf


        if lower_method == LowerMethod.Codegen:
            self.lower_model_codegen()
            self.generate_numba_model = self.generate_numba_model_code_gen
        elif lower_method == LowerMethod.Tensor:
            self.lower_model_tensor()
            self.generate_numba_model = self.generate_numba_model_tensor

        assemble_finish = time.time()
        print("Assemble time: ",assemble_finish - assemble_start)
        self.info.update({"Assemble time": assemble_finish - assemble_start})
        self.info.update({"Number of items": len(self.model_items)})
        self.info.update({"Number of variables": len(self.scope_variables)})
        self.info.update({"Number of equation scopes": len(self.equation_dict)})
        self.info.update({"Number of equations": len(self.compiled_eq)})
        self.info.update({"Solver": {}})



    def lower_model_codegen(self):
        #if len(self.gg.nodes)<100:
        #self.gg.as_graphviz('global')
        logging.info('lowering model')
        self.compiled_compute, self.var_func, self.vars_ordered_values, self.vars_ordered, self.scope_vars_vars = generate_equations(self.equations_parsed, self.eg, self.scoped_equations, self.scope_variables, self.scope_ids, self.aliases)

        if len(self.vars_ordered) > len(set(self.vars_ordered)):
            raise ValueError('non unique variables')

        self.scope_vars_ordered = [self.scope_vars_vars[v] for v in self.vars_ordered]



        self.pathed_variables_list = [[self.variables[self.scope_vars_vars[v].id].get_path_dot()]+self.variables[self.scope_vars_vars[v].id].path.path[self.system.id] for v in self.vars_ordered]

        self.pathed_variables = []
        aliases_ = {}

        for l in self.pathed_variables_list:
            self.pathed_variables.append(l[0])
            for a in l[1:]:
                aliases_[a] = l[0]

        for a, r in self.aliases.items():
            r_ = self.pathed_variables[self.vars_ordered.index(r)]
            a_ = self.variables[self.scope_vars_vars[a].id].get_path_dot()
            for aa in a_:
                #print(aa)
                aliases_[aa]=r_

        #self.aliases = {self.variables[self.scope_vars_vars[a].id].path.path[self.system.id][0]:
        #                    self.pathed_variables[self.vars_ordered.index(v)]
                            #self.variables[self.scope_vars_vars[v].id].path.path[self.system.id][0]
        #                 for a, v in self.aliases.items()}
        self.aliases = aliases_

        #generate_program(self.gg)
        #asdsad=asdsfsf
        #self.compiled_compute = generate(self.gg, self.vars_ordered_map, self.special_indcs)

        #self.compiled_compute = generate_code(self.gg, self.vars_ordered_map, ((0, self.states_end_ix),(self.states_end_ix, self.deriv_end_ix), (self.deriv_end_ix, self.mapping_end_ix)))


        self.info.update({"Solver": {}})

    def lower_model_tensor(self):
        # 3. Compute compiled_eq and compiled_eq_idxs, the latter mapping
        # self.synchronized_scope to compiled_eq (by index)
        equation_parser = Equation_Parser()
        self.compiled_eq, self.compiled_eq_idxs, self.eq_outgoing_mappings = equation_parser.parse(self)

        # 4. Create self.states_idx and self.derivatives_idx
        # Fixes each variable's var_idx (position), updates variables[].idx_in_scope
        scope_variables_flat = np.fromiter(
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



        # maps flat var_idx to scope_idx
        self.var_idx_to_scope_idx = np.full_like(scope_variables_flat, -1, np.int64)
        self.var_idx_to_scope_idx_from = np.full_like(scope_variables_flat, -1, np.int64)

        non_flat_scope_idx_from = [[] for _ in range(len(self.synchronized_scope))]
        non_flat_scope_idx = [[] for _ in range(len(self.synchronized_scope))]
        sum_idx = []
        sum_mapped = []
        sum_mapped_idx = []
        sum_mapped_idx_len = []
        self.sum_mapping = False

        def __get_mapping__idx(variable):
            if variable.mapping:
                return __get_mapping__idx(variable.mapping)
            else:
                return variable.idx_in_scope[0]

        for scope_idx, scope in enumerate(self.synchronized_scope.values()):
            for scope_var_idx, var in enumerate(scope.variables.values()):
                _from = __get_mapping__idx(self.variables[var.mapping_id]) \
                    if var.mapping_id else var.position

                self.var_idx_to_scope_idx[var.position] = scope_idx
                self.var_idx_to_scope_idx_from[_from] = scope_idx

                non_flat_scope_idx_from[scope_idx].append(_from)
                non_flat_scope_idx[scope_idx].append(var.position)
                if not var.mapping_id and var.sum_mapping_ids:
                    sum_idx.append(self.variables[var.id].idx_in_scope[0])
                    start_idx = len(sum_mapped)
                    sum_mapped += [self.variables[_var_id].idx_in_scope[0]
                                   for _var_id in var.sum_mapping_ids]
                    end_idx = len(sum_mapped)
                    sum_mapped_idx = sum_mapped_idx + list(range(start_idx, end_idx))
                    sum_mapped_idx_len.append(end_idx - start_idx)
                    self.sum_mapping = True

######################################### TODO @Artem: document these
        # non_flat_scope_idx is #scopes x  number_of variables indexing?
        # flat_scope_idx_from - rename to flat_var?
        self.non_flat_scope_idx_from = np.array(non_flat_scope_idx_from)
        self.non_flat_scope_idx = np.array(non_flat_scope_idx)

        self.flat_scope_idx_from = np.array([x for xs in self.non_flat_scope_idx_from for x in xs])
        self.flat_scope_idx = np.array([x for xs in self.non_flat_scope_idx for x in xs])

        self.sum_idx = np.array(sum_idx)
        self.sum_mapped = np.array(sum_mapped)
        self.sum_mapped_idxs_len = np.array(sum_mapped_idx_len, dtype=np.int64)
        if self.sum_mapping:
            self.sum_slice_idxs = np.array(sum_mapped_idx, dtype=np.int64)
        else:
            self.sum_slice_idxs = np.array([], dtype=np.int64)

        # eq_idx -> #variables used. Can this be deduced earlier in a more elegant way?
        self.num_vars_per_eq = np.fromiter(map(len, self.non_flat_scope_idx), np.int64)[
            np.unique(self.compiled_eq_idxs, return_index=True)[1]]
        # eq_idx -> # equation instances
        self.num_uses_per_eq = np.unique(self.compiled_eq_idxs, return_counts=True)[1]

        # float64 array of all variables' current value
        # self.flat_variables = np.array([x.value for x in self.variables.values()])
        # self.flat_variables_ids = [x.id for x in self.variables.values()]
        # self.scope_to_variables_idx = np.array([x.idx_in_scope for x in self.variables.values()])

        # (eq_idx, ind_eq_access) -> scope_variable.value

        # self.index_helper: how many of the same item type do we have?
        # max_scope_len: maximum number of variables one item can have
        # not correcly sized, as np.object
        # (eq_idx, ind_of_eq_access, var_index_in_scope) -> scope_variable.value
        self.index_helper = np.empty(len(self.synchronized_scope), np.int64)
        max_scope_len = max(map(len, self.non_flat_scope_idx_from))
        self.scope_vars_3d = np.zeros([len(self.compiled_eq), np.max(self.num_uses_per_eq), max_scope_len])

        self.length = np.array(list(map(len, self.non_flat_scope_idx)))

        _index_helper_counter = np.zeros(len(self.compiled_eq), int)
        # self.scope_vars_3d = list(map(np.empty, zip(self.num_uses_per_eq, self.num_vars_per_eq)))
        for scope_idx, (_flat_scope_idx_from, eq_idx) in enumerate(
                zip(self.non_flat_scope_idx_from, self.compiled_eq_idxs)):
            _idx = _index_helper_counter[eq_idx]
            _index_helper_counter[eq_idx] += 1
            self.index_helper[scope_idx] = _idx

            _l = self.num_vars_per_eq[eq_idx]
            self.scope_vars_3d[eq_idx][_idx, :_l] = scope_variables_flat[_flat_scope_idx_from]
        self.state_idxs_3d = self._var_idxs_to_3d_idxs(self.states_idx, False)
        self.deriv_idxs_3d = self._var_idxs_to_3d_idxs(self.derivatives_idx, False)
        _differing_idxs = self.flat_scope_idx != self.flat_scope_idx_from
        self.var_idxs_pos_3d = self._var_idxs_to_3d_idxs(np.arange(len(self.variables)), False)
        self.differing_idxs_from_flat = self.flat_scope_idx_from[_differing_idxs]
        self.differing_idxs_pos_flat = self.flat_scope_idx[_differing_idxs]


        self.differing_idxs_from_3d = self._var_idxs_to_3d_idxs(self.differing_idxs_from_flat, False)
        self.differing_idxs_pos_3d = self._var_idxs_to_3d_idxs(self.differing_idxs_pos_flat, False)
        self.sum_idxs_pos_3d = self._var_idxs_to_3d_idxs(self.sum_idx, False)
        self.sum_idxs_sum_3d = self._var_idxs_to_3d_idxs(self.sum_mapped, False)

        self.mapped_variables_array = np.zeros([self.differing_idxs_from_3d[0].shape[0], 3], dtype=np.int64)

        for i in range(self.differing_idxs_from_3d[0].shape[0]):
            self.mapped_variables_array[i] = [self.differing_idxs_from_3d[0][i], self.differing_idxs_from_3d[1][i],
                                              self.differing_idxs_from_3d[2][i]]

        # 6. Compute self.path_variables and updating var_idxs_pos_3d
        # var_idxs_pos_3d_helper shows position for var_idxs_pos_3d for variables that have
        # multiple path
        #
        var_idxs_pos_3d_helper = []
        for i, variable in enumerate(self.variables.values()):
            for path in variable.path.path[self.system.id]:
                self.path_variables.update({path: variable.value})
                var_idxs_pos_3d_helper.append(i)
        self.var_idxs_pos_3d_helper = np.array(var_idxs_pos_3d_helper, dtype=np.int64)

        # This can be done more efficiently using two num_scopes-sized view of a (num_scopes+1)-sized array
        _flat_scope_idx_slices_lengths = list(map(len, non_flat_scope_idx))
        self.flat_scope_idx_slices_end = np.cumsum(_flat_scope_idx_slices_lengths)
        self.flat_scope_idx_slices_start = np.hstack([[0], self.flat_scope_idx_slices_end[:-1]])

        assemble_finish = time.time()
        print("Assemble time: ",assemble_finish - assemble_start)
        self.info.update({"Assemble time": assemble_finish - assemble_start})
        self.info.update({"Number of items": len(self.model_items)})
        self.info.update({"Number of variables": len(self.scope_variables)})
        self.info.update({"Number of equation scopes": len(self.equation_dict)})
        self.info.update({"Number of equations": len(self.compiled_eq)})
        self.info.update({"Solver": {}})

    def _var_idxs_to_3d_idxs(self, var_idxs, _from):
        if var_idxs.size == 0:
            return (np.array([], dtype=np.int64), np.array([], dtype=np.int64)
                    , np.array([], dtype=np.int64))
        _scope_idxs = (self.var_idx_to_scope_idx_from if _from else
                       self.var_idx_to_scope_idx)[var_idxs]
        _non_flat_scope_idx = self.non_flat_scope_idx_from if _from else self.non_flat_scope_idx
        return (
            self.compiled_eq_idxs[_scope_idxs],
            self.index_helper[_scope_idxs],
            np.fromiter(itertools.starmap(list.index,
                                          zip(map(list, _non_flat_scope_idx[_scope_idxs]),
                                              var_idxs)),
                        np.int64))


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
        #return self.scope_vars_3d[self.state_idxs_3d]
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
            #print(namespace.part_of_set)
            model_namespace = ModelNamespace(namespace.tag, namespace.outgoing_mappings, item.tag, namespace.items, namespace.path, namespace.part_of_set)
            model_namespace.mappings = namespace.mappings
            model_namespace.variable_scope = namespace.variable_scope
            model_namespace.set_variables = namespace.set_variables

            equation_dict = {}
            eq_variables_ids = []
            for eq in namespace.associated_equations.values():
                equations = []
                ids = []
                for equation in eq.equations:
                    equations.append(equation)
                for vardesc in eq.variables_descriptions:
                    variable = namespace.get_variable(vardesc.tag)
                    #self.variables.update({variable.id: variable})
                    ids.append(variable.id)
                equation_dict.update({eq.tag: equations})
                eq_variables_ids.append(ids)
            model_namespace.equation_dict = equation_dict
            #model_namespace.eq_variables_ids = [[ScopeVariable(v) for v in vs] for vs in namespace.variable_scope]
            #model_namespace.variables = {v.id: ScopeVariable(v) for v in namespace.variables.shadow_dict.values()}
            model_namespace.variables = {v.id: ScopeVariable(v) for vs in namespace.variable_scope for v in vs}
            self.variables.update(model_namespace.variables)
            self.variables_set_var.update({v.id: v.set_var for vs in namespace.variable_scope for v in vs})
            #model_namespace.variables = {v.id: sv for vs, evi in zip(namespace.variable_scope, model_namespace.eq_variables_ids) for v, sv in zip(evi, vs)}
            namespaces_list.append(model_namespace)
        return namespaces_list

    # Method that generates numba_model

    def generate_numba_model_code_gen(self, start_time, number_of_timesteps):
        from numba import float64, int64
        compute = self.compiled_compute
        var_func = self.var_func
        spec = [
            ('variables', float64[:]),
            ('historian_data', float64[:,:]),
            ('historian_ix', int64),

        ]

        @jitclass(spec)
        class CompiledModel:
            def __init__(self, variables_init):

                self.variables = variables_init
                self.historian_data = np.zeros((number_of_timesteps, len(self.variables)+1), dtype=np.float64)
                #self.historian_data = np.zeros((2,2), dtype=np.float64)
                self.historian_ix = 0

            def func(self, _t, y):
                #print('diff: ', _t)
                d = compute(y)
                return d

            def historian_update(self, t):
                #print('hist update')

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
            time = self.numba_model.historian_data[:,0]
            data = {'time': time}

            for i, var in enumerate(self.pathed_variables):
                 #data.update({".".join(self.variables[var.id].path.path[self.system.id]): self.numba_model.historian_data[:,i+1]})
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
        #df['system.alias.T']

        self.aliases = aliases
        print(data.keys())

        super().__init__(data)


    def __getitem__(self, item):
        if not isinstance(item, list):
            col = self.aliases[item] if item in self.aliases else item
            return super().__getitem__(col)

        cols = [self.aliases[i] if i in self.aliases else i for i in item]


        return super().__getitem__(cols)


