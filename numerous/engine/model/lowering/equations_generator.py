from ast_builder import ASTBuilder
from numerous.engine.model.graph_representation import EdgeType
from numerous.engine.model.lowering.llvm_builder import LLVMBuilder
from numerous.engine.model.utils import NodeTypes, wrap_function, dot_dict, recurse_Attribute

from numerous.engine.model.lowering.utils import generate_code_file, Vardef, contains_dot, \
    non_unique_check

from numerous.engine.model.graph_representation.parser_ast import function_from_graph_generic, \
    compiled_function_from_graph_generic_llvm

from numerous.engine.variables import VariableType
import logging
import ast
from numba import objmode
import numpy as np

from numerous.utils.string_utils import d_u


class EquationGenerator:
    def __init__(self, filename, equation_graph, scope_variables, equations, scoped_equations, temporary_variables):
        self.filename = filename
        self.scope_variables = scope_variables
        self.set_variables = {}
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
                    if k == var.variable.id:
                        tail.update({k: v})
                        new_sv.update({var.id: var})
                    else:
                        new_sv.update({k: v})
                self.scope_variables = dict(new_sv, **tail)

        self.scoped_equations = scoped_equations
        self.temporary_variables = temporary_variables

        self.states = []

        self.deriv = []
        self.values_order = {}

        self.vars_node_id = {}
        self.scope_var_node = {}
        self.scalar_variables = {}

        self._parse_variables()

        # Sort the graph topologically to start generating code
        self.topo_sorted_nodes = equation_graph.topological_nodes()
        self.equation_graph = equation_graph.clean()

        self.number_of_states = len(self.states)
        self.number_of_derivatives = len(self.deriv)

        # Initialize llvm builder - will be a list of intermediate llvm instructions to be lowered in generate
        self.llvm = True
        if self.llvm:
            self.generated_program = LLVMBuilder(
                np.ascontiguousarray([x.value for x in self.scope_variables.values()], dtype=np.float64),
                self.values_order, self.states, self.deriv)
        else:
            self.generated_program = ASTBuilder(
                np.ascontiguousarray([x.value for x in self.scope_variables.values()], dtype=np.float64),
                self.values_order, self.states, self.deriv)

        self.mod_body = []
        # Create a kernel of assignments and calls
        self.body = []

        self.eq_vardefs = {}
        # Loop over equation functions and generate code for each equation.
        self._parse_equations(equations)

        self.all_targeted = []
        self.all_read = []
        self.all_targeted_set_vars = []
        self.all_read_set_vars = []

    def _parse_variable(self, full_tag, sv, sv_id):
        if not sv_id in self.vars_node_id:
            self.vars_node_id[full_tag] = full_tag

        if full_tag not in self.scope_var_node:
            self.scope_var_node[full_tag] = sv

        if sv.type == VariableType.STATE:
            self.states.append(self.vars_node_id[full_tag])
        elif sv.type == VariableType.DERIVATIVE:
            self.deriv.append(self.vars_node_id[full_tag])

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

    def get_external_function_name(self,ext_func):
        if self.llvm:
            return self._llvm_func_name(ext_func)
        return ext_func
    def _llvm_func_name(self, ext_func):
        return ext_func + '_llvm1.<locals>.' + ext_func + '_llvm'

    def _parse_equations(self, equations):
        logging.info('make equations for compilation')
        for eq_key, eq in equations.items():
            vardef = Vardef(llvm = self.llvm)

            func, args, target_ids = function_from_graph_generic(eq[2], eq_key, var_def_=vardef)

            eq[2].lower_graph = None
            if self.llvm:
                func_llvm, signature, args, target_ids = compiled_function_from_graph_generic_llvm(eq[2],
                                                                                                   eq_key,
                                                                                                   var_def_=Vardef(llvm=self.llvm),
                                                                                                   compiled_function=True)
                self.generated_program.add_external_function(func_llvm, signature, len(args), target_ids)
            else:
                self.generated_program.add_external_function(func, None, len(args), target_ids)

            vardef.llvm_target_ids = target_ids
            vardef.args_order = args
            self.eq_vardefs[eq_key] = vardef

    def _process_equation_node(self, n):

        eq_key = self.scoped_equations[self.equation_graph.key_map[n]]

        # Define the function to call for this eq
        ext_func = recurse_Attribute(self.equation_graph.get(n, 'func'))

        vardef = self.eq_vardefs[eq_key]

        # Find the arguments by looking for edges of arg type
        a_indcs, a_edges = list(
            self.equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val=EdgeType.ARGUMENT))
        # Determine the local arguments names
        args_local = [self.equation_graph.key_map[ae[0]] for i, ae in zip(a_indcs, a_edges) if
                      not self.equation_graph.edges_attr['arg_local'][i] == 'local']

        # Determine the local arguments names
        args_scope_var = [self.equation_graph.edges_attr['arg_local'][i] for i, ae in zip(a_indcs, a_edges) if
                          not self.equation_graph.edges_attr['arg_local'][i] == 'local']

        # Find the targets by looking for target edges
        t_indcs, t_edges = list(
            self.equation_graph.get_edges_for_node_filter(start_node=n, attr='e_type', val=EdgeType.TARGET))
        targets_local = [self.equation_graph.key_map[te[1]] for i, te in zip(t_indcs, t_edges) if
                         not self.equation_graph.edges_attr['arg_local'][i] == 'local']
        targets_scope_var = [self.equation_graph.edges_attr['arg_local'][i] for i, ae in zip(t_indcs, t_edges)
                             if
                             not self.equation_graph.edges_attr['arg_local'][i] == 'local']
        set_size = 0
        # Record targeted and read variables
        if self.equation_graph.get(n, 'vectorized'):

            # Map of scope.?? vars and set variable names
            scope_vars = {'scope.' + self.set_variables[k].tag: v for k, v in
                          zip(args_scope_var + targets_scope_var, args_local + targets_local)}

            # Put the information of args and targets in the scope_var attr of the graph node for those equation
            self.equation_graph.nodes_attr['scope_var'][n] = {'args': [scope_vars[a] for a in vardef.args],
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
            self.equation_graph.nodes_attr['scope_var'][n] = {'args': [scope_vars[a] for a in vardef.args],
                                                              'targets': [scope_vars[a] for a in vardef.targets]}
            for a in vardef.args:
                if (sva := scope_vars[a]) in self.set_variables:
                    self.all_read_set_vars.append(sva)
                else:
                    self.all_read.append(sva)

            self.all_targeted += [scope_vars[t] for t in
                                  vardef.targets]

        # Generate ast for this equation call
        args_ast = [ast.Name(id=d_u(scope_vars[a])) for a in vardef.args]
        if self.equation_graph.get(n, 'vectorized'):
            # Generate ast for targets
            if len(vardef.targets) > 1:
                targets = [ast.Tuple(
                    elts=[ast.Subscript(value=ast.Name(id=d_u(scope_vars[t])), slice=ast.Name(id='i')) for t
                          in
                          vardef.targets])]
            else:
                targets = [ast.Name(id=d_u(scope_vars[vardef.targets[0]]))]

            self.body.append(
                # For loop over items in set
                ast.For(
                    body=[ast.Assign(targets=targets, value=ast.Call(
                        func=ast.Name(id=self.scoped_equations[self.equation_graph.key_map[n]].replace('.', '_')),
                        args=[ast.Subscript(value=ast.Name(id=a), slice=ast.Index(value=ast.Name(id='i'))) for a
                              in
                              args_ast], keywords=[]))],
                    orelse=[],
                    iter=ast.Call(func=ast.Name(id='range'),
                                  args=[ast.Num(n=set_size)],
                                  keywords=[], target=ast.Name(id='i')),
                    target=ast.Name(id='i')
                )
            )

            llvm_args = []
            for t in vardef.args_order:
                llvm_args_ = []
                set_var = self.set_variables[scope_vars[t]]
                for i in range(set_var.get_size()):
                    llvm_args_.append(set_var.get_var_by_idx(i).id)
                llvm_args.append(llvm_args_)
            ##reshape to correct format
            llvm_args = [list(x) for x in zip(*llvm_args)]
            self.generated_program.add_set_call(self.get_external_function_name(ext_func), llvm_args, vardef.llvm_target_ids)


        else:

            if len(vardef.targets) > 1:
                targets = [ast.Tuple(
                    elts=[ast.Name(id=d_u(scope_vars[t])) for t in
                          vardef.targets])]
            else:
                targets = [ast.Name(id=d_u(scope_vars[vardef.targets[0]]))]

            self.body.append(ast.Assign(targets=targets, value=ast.Call(
                func=ast.Name(id=self.scoped_equations[self.equation_graph.key_map[n]].replace('.', '_')),
                args=args_ast,
                keywords=[])))

            # Generate llvm arguments
            args = [d_u(scope_vars[a]) for a in vardef.args_order]

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

                    if (mix := self.equation_graph.edges_attr['mappings'][v_ix]) == ':':
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
                            f'mapping indices not specified!{self.equation_graph.edges_attr["mappings"][v_ix]}, {self.equation_graph.key_map[t]} <- {self.equation_graph.key_map[v[0]]}')

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

            # Generate ast for the mappings

            # Utility function to make a ..+..+.. type ast from list of elts
            def add_ast_gen(elts_to_sum, op=ast.Add()):
                prev = None
                for ets in elts_to_sum:
                    if prev:
                        prev = ast.BinOp(op=op, left=prev, right=ets)
                    else:
                        prev = ets
                return prev

            map_targets = []
            map_values = []

            for t_ix, map_ in enumerate(mappings_ast_pairs):
                map_targets.append(ast.Subscript(
                    slice=ast.Index(value=ast.Num(n=t_ix)), value=ast.Name(id=d_u(t_sv.id))))
                if len(map_) > 0:
                    map_val_list = [ast.Subscript(
                        slice=ast.Index(value=ast.Num(n=v_ix)),
                        value=ast.Name(id=d_u(v_target))) if not v_ix is None else ast.Name(id=d_u(v_target))
                                    for
                                    v_target, v_indcs in map_ for v_ix in v_indcs]
                else:
                    map_val_list = [ast.Num(n=0)]

                map_values.append(map_val_list)

            self.body.append(ast.Assign(targets=[ast.Tuple(elts=map_targets)],
                                        value=ast.Tuple(elts=[add_ast_gen(mv) for mv in map_values])))

            if len(mappings[':']) > 0:
                # Mappings of full set vars to the target
                prev = None

                for mcolon in mappings[':']:

                    if prev:
                        prev = ast.BinOp(left=prev, right=ast.Name(id=d_u(mcolon)), op=ast.Add())
                    else:
                        prev = ast.Name(id=d_u(mcolon))
                if len(mappings['ix']) > 0:
                    self.body.append(
                        ast.AugAssign(target=ast.Name(id=d_u(t_sv.id)), value=prev, op=ast.Add()))
                else:
                    self.body.append(ast.Assign(targets=[ast.Name(id=d_u(t_sv.id))], value=prev))

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
                # for v1 in v:
                #     print(f"Mapping  {self.scope_variables[v1].path.primary_path}"
                #           f" to {self.scope_variables[k].path.primary_path} ")
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

                maps = self.equation_graph.edges_attr['mappings'][vi]

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

            # Generate ast
            if self.equation_graph.get(t, 'is_set_var'):
                map_targs = ast.Tuple(
                    elts=[ast.Subscript(value=ast.Name(id=d_u(target_var)), slice=ast.Index(value=ast.Num(n=i)))
                          for
                          i, _ in enumerate(target_indcs_map) if len(_) > 0])
            else:
                map_targs = ast.Name(id=d_u(target_var))

            map_values = []
            for values in target_indcs_map:

                prev = None
                for v in values:
                    v_ = ast.Name(id=d_u(self.equation_graph.key_map[v[0]])) if v[1] is None else ast.Subscript(
                        value=ast.Name(id=d_u(self.equation_graph.key_map[v[0]])),
                        slice=ast.Index(value=ast.Num(n=v[1])))
                    if prev:
                        prev = ast.BinOp(op=ast.Add(), left=v_, right=prev)
                    else:
                        prev = v_

                map_values.append(prev)

            assign = ast.Assign(targets=[map_targs],
                                value=ast.Tuple(elts=map_values) if len(map_values) > 1 else map_values[0])
            self.body.append(assign)

            # Generate llvm
            for values in target_indcs_map:
                for v in values:
                    var_name = d_u(self.equation_graph.key_map[v[0]])
                    if var_name in self.scope_variables:
                        print(f"Mapping  {self.scope_variables[var_name].path.primary_path}"
                              f" to {self.scope_variables[target_var].path.primary_path} ")
                        self.generated_program.add_mapping([var_name], [target_var])
                    else:
                        if var_name in self.set_variables:
                            print(f"Mapping  {self.set_variables[var_name].get_var_by_idx(v[1]).path.primary_path} "
                                  f"to {self.scope_variables[target_var].path.primary_path} ")
                            self.generated_program.add_mapping(
                                [self.set_variables[var_name].get_var_by_idx(v[1]).id], [target_var])
                        else:
                            raise ValueError(f'Variable  {var_name} mapping not found')

    def generate_equations(self, use_llvm=True):
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
        # self.generated_program.generate()
        if use_llvm:
            logging.info('generating llvm')
            diff, var_func, var_write = self.generated_program.generate("test_listing.txt", save_opt=True)

            return diff, var_func, var_write, self.values_order, self.scope_variables, np.array(state_idx,
                                                                                                dtype=np.int64), np.array(
                deriv_idx, dtype=np.int64)

        #
        # Update maps between scope variables
        for sv_id, sv in self.scope_variables.items():
            full_tag = d_u(sv.get_path_dot())
            if not sv_id in self.vars_node_id:
                self.vars_node_id[sv_id] = full_tag
            if full_tag not in self.scope_var_node:
                self.scope_var_node[full_tag] = sv
        #
        #
        # Only variables that are not targeted should be in read
        all_read_set_vars = set(self.all_read_set_vars).difference(self.all_targeted_set_vars)

        # Unroll all scalar variables from set variables
        all_read_scalars_from_set = []
        for arsv in all_read_set_vars:
            a = self.set_variables[arsv]
            all_read_scalars_from_set += [v.get_path_dot() for v in a.variables.values()]

        all_read_scalars_from_set = set(all_read_scalars_from_set)
        all_read_scalars_from_set_dash = [d_u(ar) for ar in all_read_scalars_from_set]

        all_read = set(self.all_read)

        # Check for overlap of read and read in set - not allowed!
        r_setvar_r_overlap = all_read_scalars_from_set.intersection(all_read)
        if len(r_setvar_r_overlap) > 0:
            raise ValueError(f"Overlap between read vars and read vars in set {r_setvar_r_overlap}")

        all_read_dash = [d_u(ar) for ar in all_read]
        contains_dot(all_read_dash)

        all_targeted_scalars_from_set = []
        for arsv in self.all_targeted_set_vars:
            a = self.set_variables[arsv]
            all_targeted_scalars_from_set += [v.get_path_dot() for v in a.variables.values()]

        all_targeted_scalars_from_set = set(all_targeted_scalars_from_set)
        all_targeted_scalars_from_set_dash = [d_u(ar) for ar in all_targeted_scalars_from_set]

        all_targeted = set(self.all_targeted)

        all_targeted_dash = [d_u(ar) for ar in all_targeted]

        contains_dot(all_targeted_dash)
        all_must_init = set(all_read_dash).difference(all_targeted_dash)

        states_dash = [d_u(s) for s in self.states]
        vars_init = states_dash.copy()

        all_read_scalars_from_set_dash = list(set(all_read_scalars_from_set_dash).difference(vars_init))

        vars_init += list(all_must_init.difference(vars_init))

        leninit = len(vars_init + all_read_scalars_from_set_dash)

        non_unique_check('deriv: ', self.deriv)
        non_unique_check('all targeted: ', all_targeted_dash)

        vars_update = [d_u(d) for d in self.deriv.copy()]

        all_targeted_scalars_from_set_dash = list(set(all_targeted_scalars_from_set_dash).difference(vars_update))

        vars_update += [at for at in all_targeted_dash if at not in vars_update]

        for s, d in zip(self.states, self.deriv):
            if not d[:-4] == s:
                # print(d, ' ', s)
                raise IndexError('unsorted derivs')

        variables = vars_init + all_read_scalars_from_set_dash + vars_update + all_targeted_scalars_from_set_dash

        variables += set([d_u(sv.get_path_dot()) for sv in self.scope_variables.values()]).difference(variables)

        variables_dot = [self.scope_var_node[v].get_path_dot() for v in variables]

        variables_values = np.array([self.scope_var_node[v].value for v in variables], dtype=np.float64)

        non_unique_check('initialized vars', vars_init)
        len_vars_init_ = len(vars_init + all_read_scalars_from_set_dash)
        vars_init += self.deriv
        non_unique_check('updated vars', vars_update)

        non_unique_check('variables', variables)

        # Generate ast for defining local variables and export results
        state_vars = [None] * self.number_of_states
        body_init_set_var = []
        for rsv in all_read_set_vars:

            # Find indices in variables
            read_scalars = [d_u(v[1].get_path_dot()) for v in self.set_variables[rsv]]
            read_scalars_var_ix = [variables.index(r) for r in read_scalars]

            body_init_set_var.append(ast.Assign(targets=[ast.Name(id=d_u(rsv))],
                                                value=ast.Subscript(value=ast.Name(id='variables'), slice=ast.Index(
                                                    value=ast.List(
                                                        elts=[ast.Num(n=ix) for ix in read_scalars_var_ix])))))

            state_vars_ = [(v, states_dash.index(v)) for v in read_scalars if v in states_dash]

            for v, ix in state_vars_:
                state_vars[ix] = ast.Subscript(value=ast.Name(id=d_u(rsv)),
                                               slice=ast.Index(value=ast.Num(n=read_scalars.index(v))))

        for i, s in enumerate(states_dash):
            if state_vars[i] is None:
                state_vars[i] = ast.Name(id=s)

        body_init_set_var.append(ast.Assign(value=ast.Name(id='y'), targets=[ast.Tuple(elts=state_vars)]))

        for tsv in self.all_targeted_set_vars:
            body_init_set_var.append(ast.Assign(targets=[ast.Name(id=d_u(tsv))],
                                                value=ast.Call(
                                                    func=ast.Attribute(attr='empty', value=ast.Name(id='np')),
                                                    args=[ast.Num(n=len(self.set_variables[tsv]))], keywords=[]
                                                )))

        self.body = body_init_set_var + [
            ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id=d_u(i)) for i in vars_init[self.number_of_states:]])],
                       value=ast.Subscript(
                           slice=ast.Slice(lower=ast.Num(n=self.number_of_states), upper=ast.Num(n=len(vars_init)),
                                           step=None),
                           value=ast.Name(id='variables'))),
        ] + self.body

        # Add code for updating variables

        elts_vu = [
            (ast.Subscript(value=ast.Name(id=d_u(svn_.set_var)), slice=ast.Index(ast.Num(n=svn_.set_var_ix))) if (
                svn_ := self.scope_var_node[u]).set_var else ast.Name(id=d_u(u))) for i, u in enumerate(vars_update)]

        self.body.append(ast.Assign(targets=[ast.Subscript(
            slice=ast.Slice(lower=ast.Num(n=len_vars_init_), upper=ast.Num(n=len_vars_init_ + len(vars_update)),
                            step=None),
            value=ast.Name(id='variables'))],
            value=ast.Tuple(elts=elts_vu)))

        # Add code for updating derivatives
        self.body.append(ast.Assign(value=ast.Tuple(elts=[
            ast.Subscript(value=ast.Name(id=d_u(svn_.set_var)), slice=ast.Index(ast.Num(n=svn_.set_var_ix))) if (
                svn_ := self.scope_var_node[u]).set_var else ast.Name(id=d_u(u)) for u in self.states
        ]), targets=[
            ast.Subscript(slice=ast.Slice(lower=ast.Num(n=0), upper=ast.Num(n=self.number_of_states), step=None),
                          value=ast.Name(id='variables'))]))

        self.body.append(ast.Return(
            value=ast.Subscript(
                slice=ast.Slice(lower=ast.Num(n=leninit), upper=ast.Num(n=leninit + self.number_of_derivatives),
                                step=None),
                value=ast.Name(id='variables'))))
        kernel_args = dot_dict(args=[ast.Name(id='variables'), ast.Name(id='y')], vararg=None, defaults=[], kwarg=None)


        self.mod_body.append(wrap_function('kernel_nojit', self.body, decorators=[], args=kernel_args))

        generate_code_file(self.mod_body, self.filename)
        logging.info('compiling...')

        import timeit
        print('Compile time: ', timeit.timeit(
            lambda: exec('from kernel import *', globals()), number=1))

        from numba import njit

        ###TESTS####
        y = variables_values[:self.number_of_derivatives].astype(np.float64)

        from time import time
        N = 10000

        # TODO: Use this code to define a benchmark of llvm
        # @njit('float64[:](float64[:], int64)')
        # def diff_bench_llvm(y, N):

        #    for i in range(N):
        #        derivatives = diff_llvm(y)

        #    return derivatives

        # tic = time()
        # derivs_llvm = diff_bench_llvm(y, N)
        # toc = time()
        # llvm_vars = var_func(0)
        # print('llvm derivs: ', list(zip(deriv, derivs_llvm)))
        # print('llvm vars: ', list(zip(variables, var_func(0))))
        # print(f'Exe time llvm - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)

        N = 5
        if not skip_kernel:
            def test_kernel_nojit(variables, y):
                for i in range(N):
                    deriv = kernel_nojit(variables, y)
                    # print(deriv)
                return deriv

            # print(y)
            tic = time()
            deriv_no_jot = test_kernel_nojit(variables_values, y)
            toc = time()

            class AssemlbedModel():
                def __init__(self, vars, vals):
                    self.variables = vars
                    self.init_vals = vals

                    @njit
                    def diff(y):
                        with objmode(derivs='float64[:]'):  # annotate return type
                            # this region is executed by object-mode.'
                            # print(y)
                            derivs = self.diff__(y)
                            # print(derivs)
                        return derivs.copy()

                    self.diff = diff

                    @njit
                    def var_func(i):
                        with objmode(vrs='float64[:]'):  # annotate return type
                            # this region is executed by object-mode.
                            vrs = self.vars__()
                            # print(vrs)
                        return vrs.copy()

                    self.var_func = var_func

                def diff__(self, y):
                    return kernel_nojit(self.init_vals, y)

                def vars__(self):
                    # for v, vv in zip(self.variables, self.init_vals):
                    #    print(v,': ',vv)
                    return self.init_vals

            am = AssemlbedModel(variables, variables_values)
            diff_ = am.diff
            var_func_ = am.var_func

            # print(deriv_no_jot)
            print(f'Exe time flat no jit - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)

            print('no jit derivs: ', list(zip(self.deriv, deriv_no_jot)))
            print('no jit vars: ', list(zip(variables, am.var_func(0))))

            # TODO: This code can be moved to a test to check if results of llvm and regular kernel is the same
            # print('var diff')
            # for k, v_n, v_llvm in zip(variables, am.var_func(0), var_func(0)):
            #    print(k,': ',v_n,' ',v_llvm,' diff: ', v_n-v_llvm)

            # print('deriv diff')
            # for k, v_n, v_llvm in zip(deriv, deriv_no_jot, derivs_llvm):
            #    if abs(v_n) >1e-20:
            ##        rel_diff = (v_n - v_llvm) / v_n
            #    else:
            #        rel_diff = 0

            #    print(k,': ',v_n,' ',v_llvm,' rel diff: ', rel_diff)
            #    if rel_diff>0.001:
            #        raise ValueError(f'Arg {k}, {v_n}, {v_llvm}, {rel_diff}')

            # print('Exe time kernel nojit timeit: ', timeit.timeit(
            #    lambda: kernel_nojit(variables_values, y), number=N) / N)

            N = 10000

            # @njit('void(float64[:], float64[:])')
            # def test_kernel(variables, y):
            #    for i in range(N):
            #        kernel(variables, y)

            print('First kernel call results: ')
            # print(kernel(variables_values, y))

            tic = time()
            # test_kernel(variables_values, y)
            toc = time()

            print(f'Exe time flat - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)
            # print('Exe time kernel timeit: ', timeit.timeit(
            #    lambda: kernel(variables_values, y), number=N) / N)
        count = 0
        # for v, v_llvm, v_kernel in zip(variables, llvm_vars, variables_values):
        #    err = abs(v_llvm - v_kernel)/abs(v_llvm + v_kernel)*2
        #    if err>1e-3:
        #        print(v,': ',v_llvm,' ',v_kernel, ' ', err)
        # count+=1
        # print(count)

        # sdfsdf=sdfsdf
        ###TEST PROGRAM
        # spec = [
        #    ('program', int64[:, :]),
        #    ('indices', int64[:]),

        # ]
        N = 10000
        """
        from numba.experimental import jitclass
        @jitclass(spec)
        class DiffProgram:
            def __init__(self, program, indices):
                self.program = program
                self.indices = indices

            def diff(self, variables, y):
                return diff(variables, y, self.program, self.indices)

            def test(self, variables, y):
                for i in range(N):
                    self.diff(variables, y)

        #dp = DiffProgram(np.array(program, np.int64), np.array(indices, np.int64))
        """

        print('First prgram call results: ')
        # print(dp.diff(variables_values, y))

        # tic = time()
        # dp.test(variables_values, y)
        # toc = time()

        # print(f'Exe time program - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)

        # print('Exe time program timeit: ', timeit.timeit(
        #        lambda: dp.diff(variables_values, y), number=N)/N)
        for v, vv in zip(variables, variables_values):
            print(v, ': ', vv)

        return diff_, var_func_, variables_values, variables_dot, self.scope_var_node
