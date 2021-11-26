import logging
import os
import sys
import time
import types

import numpy as np

from numerous.engine.model.ast_parser.equation_form_graph import function_from_graph_generic, \
    compiled_function_from_graph_generic_llvm
from numerous.engine.model.lowering.ast_builder import ASTBuilder
from numerous.engine.model.graph_representation import EdgeType

from numerous.engine.model.lowering.llvm_builder import LLVMBuilder
from numerous.engine.model.lowering.utils import Vardef
from numerous.engine.model.utils import NodeTypes, recurse_Attribute
from numerous.engine.variables import VariableType
from numerous.utils.string_utils import d_u


class EquationGenerator:
    def __init__(self, filename, equation_graph, scope_variables, equations, scoped_equations,
                 temporary_variables, system_tag="", use_llvm=True, imports=None, eq_used=[]):
        self.filename = filename
        self.imports = imports
        self.system_tag = system_tag
        self.scope_variables = scope_variables
        self.set_variables = {}
        self.states = []
        self.deriv = []

        for ix, (sv_id, sv) in enumerate(self.scope_variables.items()):
            full_tag = d_u(sv.id)
            if sv.type == VariableType.STATE:
                self.states.append(full_tag)
            elif sv.type == VariableType.DERIVATIVE:
                self.deriv.append(full_tag)

        for k, var in temporary_variables.items():
            if var.type == VariableType.TMP_PARAMETER_SET:
                self.set_variables.update({k: var})
                new_sv = {}
                tail = {}
                for k, v in self.scope_variables.items():
                    if k in var.set_var.variables:
                        tail.update({k: v})
                        new_sv.update({var.tmp_vars[v.set_var_ix].id: var.tmp_vars[v.set_var_ix]})
                    else:
                        new_sv.update({k: v})
                self.scope_variables = dict(new_sv, **tail)
            if var.type == VariableType.TMP_PARAMETER:
                new_sv = {}
                tail = {}
                for k, v in self.scope_variables.items():
                    if k == var.scope_var_id:
                        tail.update({k: v})
                        new_sv.update({var.id: var})
                    else:
                        new_sv.update({k: v})
                self.scope_variables = dict(new_sv, **tail)

        self.scoped_equations = scoped_equations
        self.temporary_variables = temporary_variables

        self.values_order = {}

        self.scope_var_node = {}
        self.scalar_variables = {}

        self._parse_variables()

        # Sort the graph topologically to start generating code
        self.topo_sorted_nodes = equation_graph.topological_nodes()
        self.equation_graph = equation_graph.clean()

        self.number_of_states = len(self.states)
        self.number_of_derivatives = len(self.deriv)

        # Initialize llvm builder - will be a list of intermediate llvm instructions to be lowered in generate
        self.llvm = use_llvm
        if self.llvm:
            self.generated_program = LLVMBuilder(
                np.ascontiguousarray([x.value for x in self.scope_variables.values()], dtype=np.float64),
                self.values_order, self.states, self.deriv, system_tag=self.system_tag)
        else:
            self.generated_program = ASTBuilder(
                np.ascontiguousarray([x.value for x in self.scope_variables.values()], dtype=np.float64),
                self.values_order, self.states, self.deriv, system_tag=self.system_tag)

        self.mod_body = []
        # Create a kernel of assignments and calls

        self.eq_vardefs = {}
        # Loop over equation functions and generate code for each equation.
        print('_parse_eq: ')
        for e in equations:
            print(e)

        used_eq = {}
        print(':::')
        print(eq_used)

        for eq_key, eq in equations.items():
            #print(eq[0])
            if eq_key in eq_used:
                used_eq[eq_key]=eq
        self._parse_equations(used_eq)

        self.all_targeted = []
        self.all_read = []
        self.all_targeted_set_vars = []
        self.all_read_set_vars = []

    def _parse_variable(self, full_tag, sv, sv_id):

        if full_tag not in self.scope_var_node:
            self.scope_var_node[full_tag] = sv

        # If a scope_variable is part of a set it should be referenced alone
        if sv.set_var:
            if not sv.set_var.id in self.set_variables:
                self.set_variables[sv.set_var.id] = sv.set_var
        else:
            self.scalar_variables[full_tag] = sv

    def _parse_variables(self):
        for ix, (sv_id, sv) in enumerate(self.scope_variables.items()):
            full_tag = d_u(sv.id)
            self.values_order[full_tag] = ix
            self._parse_variable(full_tag, sv, sv_id)
        for ix, (sv_id, sv) in enumerate(self.temporary_variables.items()):
            full_tag = d_u(sv.id)
            self._parse_variable(full_tag, sv, sv_id)

    def get_external_function_name(self, ext_func):
        if self.llvm:
            return self._llvm_func_name(ext_func)
        return ext_func

    def _llvm_func_name(self, ext_func):
        return ext_func + '_llvm1.<locals>.' + ext_func + '_llvm'

    def _parse_equations(self, equations):
        logging.info('make equations for compilation')


        for eq_key, eq in equations.items():
            vardef = Vardef(llvm=self.llvm)

            eq[2].lower_graph = None
            if self.llvm:
                func_llvm, signature, args, target_ids = compiled_function_from_graph_generic_llvm(
                    eq[2],
                    imports=self.imports,
                    var_def_=Vardef(llvm=self.llvm),
                    compiled_function=True,
                    replacements=eq[4] if len(eq) > 4 else {},
                    replace_name=eq_key
                )
                self.generated_program.add_external_function(func_llvm, signature, len(args), target_ids)
            else:
                func, args, target_ids = function_from_graph_generic(eq[2],
                                                                     var_def_=vardef, arg_metadata=eq[2].arg_metadata)
                self.generated_program.add_external_function(func, None, len(args), target_ids)

            vardef.llvm_target_ids = target_ids
            vardef.args_order = args
            self.eq_vardefs[eq_key] = vardef

    def search_in_item_scope(self, var_id, item_id):
        for var in self.scope_variables.values():
            ##TODO add namespacecheck
            if var.item.id == item_id and var.tag == self.scope_variables[var_id].tag:
                return var.id
        raise ValueError("No variable found for id {}", var_id)

    def _process_equation_node(self, n):

        eq_key = self.scoped_equations[self.equation_graph.key_map[n]]

        # Define the function to call for this eq
        ext_func = recurse_Attribute(self.equation_graph.get(n, 'func'))
        item_id = self.equation_graph.get(n, 'item_id')

        vardef = self.eq_vardefs[eq_key]

        # Find the arguments by looking for edges of arg type
        a_indcs, a_edges = list(
            self.equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.ARGUMENT))
        # Determine the local arguments names
        args_local = [self.equation_graph.key_map[ae[0]] for i, ae in zip(a_indcs, a_edges) if
                      not self.equation_graph.edges_c[i].arg_local == 'local']

        # Determine the local arguments names
        args_scope_var = [self.equation_graph.edges_c[i].arg_local for i, ae in zip(a_indcs, a_edges) if
                          not self.equation_graph.edges_c[i].arg_local == 'local']

        # Find the targets by looking for target edges
        t_indcs, t_edges = list(
            self.equation_graph.get_edges_for_node_filter(start_node=n, attr='e_type', val=EdgeType.TARGET))
        targets_local = [self.equation_graph.key_map[te[1]] for i, te in zip(t_indcs, t_edges) if
                         not self.equation_graph.edges_c[i].arg_local == 'local']
        targets_scope_var = [self.equation_graph.edges_c[i].arg_local for i, ae in zip(t_indcs, t_edges)
                             if
                             not self.equation_graph.edges_c[i].arg_local == 'local']
        set_size = 0
        # Record targeted and read variables
        if self.equation_graph.get(n, 'vectorized'):

            # Map of scope.?? vars and set variable names
            scope_vars = {'scope.' + self.set_variables[k].tag: v for k, v in
                          zip(args_scope_var + targets_scope_var, args_local + targets_local)}

            # Put the information of args and targets in the scope_var attr of the graph node for those equation
            self.equation_graph.nodes[n].scope_var = {'args': [scope_vars[a] for a in vardef.args],
                                                      'targets': [scope_vars[a] for a in vardef.targets]}
            # Record all targeted variables
            for t in vardef.targets:
                self.all_targeted_set_vars.append(scope_vars[t])
                ##TODO check that they all the same size
                set_size = self.set_variables[scope_vars[t]].get_size()
            # Record all read variables
            for a in vardef.args:
                self.all_read_set_vars.append(scope_vars[a])
        else:
            # Map of scope.?? vars and global-scope variable names
            scope_vars = {'scope.' + self.scope_variables[k].tag: v for k, v in
                          zip(args_scope_var + targets_scope_var, args_local + targets_local)}

            # Put the information of args and targets in the scope_var attr of the graph node for those equation
            self.equation_graph.nodes[n].scope_var = {'args': [scope_vars[a] for a in vardef.args],
                                                      'targets': [scope_vars[a] for a in vardef.targets]}
            for a in vardef.args:
                if (sva := scope_vars[a]) in self.set_variables:
                    self.all_read_set_vars.append(sva)
                else:
                    self.all_read.append(sva)

            self.all_targeted += [scope_vars[t] for t in
                                  vardef.targets]

        # Generate ast for this equation callcompiled_function_from_graph_generic_llvm
        if self.equation_graph.get(n, 'vectorized'):

            llvm_args = []
            for t in vardef.args_order:
                llvm_args_ = []
                set_var = self.set_variables[scope_vars[t]]
                for i in range(set_var.get_size()):
                    llvm_args_.append(set_var.get_var_by_idx(i).id)
                llvm_args.append(llvm_args_)
            ##reshape to correct format
            llvm_args = [list(x) for x in zip(*llvm_args)]
            self.generated_program.add_set_call(self.get_external_function_name(ext_func), llvm_args,
                                                vardef.llvm_target_ids)
        else:
            # Generate llvm arguments
            args = []

            for a in vardef.args_order:
                if a in scope_vars:
                    args.append(d_u(scope_vars[a]))
                else:
                    args.append(self.search_in_item_scope(a, item_id))

            # Add this eq to the llvm_program
            self.generated_program.add_call(self.get_external_function_name(ext_func), args, vardef.llvm_target_ids)

    def _process_sum_node(self, n):
        t_indcs, target_edges = list(
            self.equation_graph.get_edges_for_node_filter(start_node=n, attr='e_type', val=EdgeType.TARGET))
        v_indcs, value_edges = list(
            self.equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.VALUE))

        # assume single target
        if lte := len(target_edges) != 1:
            raise ValueError(f'Wrong length of target edges - must be 1 but is {lte}')
        t = target_edges[0][1]

        # If the target is a set variable
        if (t_sv := self.equation_graph.get(t, 'scope_var')).size > 0:

            self.all_targeted_set_vars.append(self.equation_graph.key_map[t])

            l_mapping = t_sv.size
            mappings = {':': [], 'ix': []}

            # make a list of assignments to each index in t
            for v_ix, v in zip(v_indcs, value_edges):
                if (nt := self.equation_graph.get(v[0], 'node_type')) == NodeTypes.VAR or nt == NodeTypes.TMP:

                    if (mix := self.equation_graph.edges_c[v_ix].mappings) == ':':
                        mappings[':'].append(self.equation_graph.key_map[v[0]])

                    elif isinstance(mix, list):
                        sums = {}
                        for m in mix:
                            if not m[1] in sums:
                                sums[m[1]] = []
                            sums[m[1]].append(m[0])
                        mappings['ix'].append((self.equation_graph.key_map[v[0]], sums))

                    else:
                        raise ValueError(
                            f'mapping indices not specified!{self.equation_graph.edges_c[v_ix].mappings}, {self.equation_graph.key_map[t]} <- {self.equation_graph.key_map[v[0]]}')

                else:
                    raise ValueError(f'this must be a mistake {self.equation_graph.key_map[v[0]]}')

            mappings_ast_pairs = [
                                     []] * l_mapping  # To be list of tuples of var and target for each indix in target

            # process specific index mappings
            for m_ix in mappings['ix']:

                # m_ix[0] is a variable mapped to the current set variable
                # m_ix[1] is a dict:
                # Keys which are indices to this set variable.
                # Values which are indices to the variable mapped to this set variable
                from_ = m_ix[0]
                # m_ix1_keys = list(m_ix[1].keys())

                # loop over all indices in target
                for target_ix, value_ix in m_ix[1].items():
                    mappings_ast_pairs[target_ix].append((from_, value_ix))

            # Generate llvm
            mappings_llvm = {}
            if len(mappings[':']) > 0:
                for set_variable in mappings[':']:
                    for k in range(self.set_variables[set_variable].size):
                        target_var_name = self.set_variables[t_sv.id].get_var_by_idx(k).id
                        mappings_llvm[target_var_name] = [self.set_variables[set_variable].get_var_by_idx(k).id]
            for m_ix in mappings['ix']:
                for k, v in m_ix[1].items():
                    target_var_name = self.set_variables[t_sv.id].get_var_by_idx(k).id
                    if target_var_name not in mappings_llvm.keys():
                        mappings_llvm[target_var_name] = []
                    for el in v:
                        if m_ix[0] in self.set_variables:
                            mappings_llvm[target_var_name].append(self.set_variables[m_ix[0]].get_var_by_idx(el).id)
                        else:
                            if m_ix[0] in self.scope_variables:
                                mappings_llvm[target_var_name].append(self.scope_variables[m_ix[0]].id)
                            else:
                                raise ValueError(f'Variable  {m_ix[0]} mapping not found')
            for k, v in mappings_llvm.items():
                self.generated_program.add_mapping(v, [k])
        else:
            # Register targeted variables
            if is_set_var := self.equation_graph.get(t, attr='is_set_var'):
                self.all_targeted_set_vars.append(self.equation_graph.key_map[t])
            else:

                self.all_targeted.append(self.equation_graph.key_map[t])

            target_indcs_map = [[] for i in
                                range(len(
                                    self.set_variables[self.equation_graph.key_map[t]]))] if is_set_var else [
                []]

            for v, vi in zip(value_edges, v_indcs):
                if self.equation_graph.get(v[0], 'is_set_var'):
                    self.all_read_set_vars.append(self.equation_graph.key_map[v[0]])

                else:
                    self.all_read.append(self.equation_graph.key_map[v[0]])

                maps = self.equation_graph.edges_c[vi].mappings

                if maps == ':':
                    if self.equation_graph.key_map[t] in self.set_variables:
                        for mi in range(len(self.set_variables[self.equation_graph.key_map[t]])):
                            target_indcs_map[mi].append((v[0], mi))
                    else:
                        target_indcs_map[0].append((v[0], None))
                else:
                    for mi in maps:
                        target_indcs_map[mi[1] if mi[1] else 0].append((v[0], mi[0]))

            target_var = self.equation_graph.key_map[t]

            # Generate llvm/ast
            mapping_dict = {}
            for values in target_indcs_map:
                for v in values:
                    var_name = d_u(self.equation_graph.key_map[v[0]])
                    if var_name in self.scope_variables:
                        if target_var in mapping_dict:
                            mapping_dict[target_var].append(var_name)
                        else:
                            mapping_dict[target_var] = [var_name]
                    else:
                        if var_name in self.set_variables:
                            if var_name in mapping_dict:
                                mapping_dict[target_var].append(self.set_variables[var_name].get_var_by_idx(v[1]).id)
                            else:
                                mapping_dict[target_var] = [self.set_variables[var_name].get_var_by_idx(v[1]).id]
                        else:
                            raise ValueError(f'Variable  {var_name} mapping not found')
            for k, v in mapping_dict.items():
                self.generated_program.add_mapping(v, [k])

    def generate_equations(self, save_to_file=False):
        logging.info('Generate kernel')
        # Generate the ast for the python kernel
        for n in self.topo_sorted_nodes:
            # Add the equation calls
            if (nt := self.equation_graph.get(n, 'node_type')) == NodeTypes.EQUATION:
                self._process_equation_node(n)
            # # Add the sum statements
            elif nt == NodeTypes.SUM:
                self._process_sum_node(n)

            elif nt == NodeTypes.VAR or nt == NodeTypes.TMP:
                pass
            else:
                raise ValueError('Unused node: ', self.equation_graph.key_map[n])

        deriv_idx = []
        state_idx = []
        for k, v in self.values_order.items():
            if k in self.deriv:
                deriv_idx.append(v)
            if k in self.states:
                state_idx.append(v)
        if self.llvm:
            logging.info('generating llvm')
            diff, var_func, var_write = self.generated_program.generate(imports=self.imports,
                                                                        system_tag=self.system_tag,
                                                                        save_to_file=save_to_file)

            return diff, var_func, var_write, self.values_order, self.scope_variables, np.array(state_idx,
                                                                                                dtype=np.int64), \
                   np.array(
                       deriv_idx, dtype=np.int64)
        else:
            global_kernel, var_func, var_write = self.generated_program.generate(self.imports,
                                                                                 system_tag=self.system_tag,
                                                                                 save_to_file=save_to_file)
            return global_kernel, var_func, var_write, self.values_order, self.scope_variables, \
                   np.array(state_idx, dtype=np.int64), np.array(deriv_idx, dtype=np.int64)
