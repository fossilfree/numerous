from numerous.engine.model.graph import Graph
from numerous.engine.model.utils import NodeTypes, recurse_Attribute, wrap_function, dot_dict, generate_code_file
from numerous.engine.model.parser_ast import attr_ast, function_from_graph_generic, function_from_graph_generic_llvm#, EquationNode, EquationEdge
from numerous.engine.variables import VariableType
from numerous.engine.model.generate_program import generate_program
import logging
import ast
from numba import njit, objmode
import numpy as np

def d_u(str_):
    return str_.replace('.','_')

class Vardef:
    def __init__(self):
        self.vars_inds_map = []
        self.targets = []
        self.args = []

    def format(self, var):
        return ast.Name(id=var.replace('scope.', 's_'))

    def var_def(self, var, read=True):
        if not var in self.vars_inds_map:
            self.vars_inds_map.append(var)
        if read and 'scope.' in var:
            if var not in self.targets and var not in self.args:
                self.args.append(var)
        elif 'scope.' in var:

            if var not in self.targets:
                self.targets.append(var)

        return self.format(var)

    def get_args(self, form=True):
        if form:
            return [self.format(a) for a in self.args]
        else:
            return self.args

    def get_targets(self, form=True):
        if form:
            return [self.format(a) for a in self.targets]
        else:
            return self.targets

class Vardef_llvm:
    def __init__(self):
        self.vars_inds_map = []
        self.targets = []
        self.args = []

    def format(self, var):
        return ast.Name(id=var.replace('scope.', 's_'))

    def format_target(self, var):
        return ast.Subscript(slice=ast.Index(value=ast.Num(n=0)), value=ast.Call(args=[ast.Name(id=var.replace('scope.', 's_')), ast.Tuple(elts=[ast.Num(n=1)])], func=ast.Name(id='carray'), keywords=[]))


    def var_def(self, var, read=True):
        if not var in self.vars_inds_map:
            self.vars_inds_map.append(var)
        if read and 'scope.' in var:
            if var not in self.targets and var not in self.args:
                self.args.append(var)
        elif 'scope.' in var:

            if var not in self.targets:
                self.targets.append(var)

        if var in self.targets:
            return self.format_target(var)
        else:
            return self.format(var)

    def get_args(self, form=True):
        if form:
            return [self.format(a) for a in self.args]
        else:
            return self.args

    def get_targets(self, form=True):
        if form:
            return [self.format(a) for a in self.targets]
        else:
            return self.targets

class SumCount:
    def __init__(self):
        self.count = -1

    def get_sum(self):
        self.count += 1
        return  f"sum_{self.count}"

new_sum = SumCount().get_sum

def visit_assign_value(target, value, nodes_map, equation_graph):
    if equation_graph.get(value, 'node_type') == NodeTypes.OP:

        ix_left, left_edges = equation_graph.get_edges_for_node_filter(end_node=value, attr='e_type', val='left')

        left_edge = left_edges[0]
        left = left_edge[0]

        visit_assign_value(target, left, nodes_map, equation_graph)
        equation_graph.remove_edge(ix_left[0])

        ix_right, right_edges = equation_graph.get_edges_for_node_filter(end_node=value, attr='e_type', val='right')
        right_edge = right_edges[0]

        right = right_edge[0]

        visit_assign_value(target, right, nodes_map, equation_graph)
        equation_graph.remove_edge(ix_right[0])

        equation_graph.remove_node(value)

    else:

        target.append(value)


import uuid

class TemporaryVar():
    tmp_var_counter = 0

    def __init__(self, svi, tmp_label):
        TemporaryVar.tmp_var_counter+=1
        self.id = 'tmp_var_'+str(TemporaryVar.tmp_var_counter) if svi.set_var else d_u(tmp_label)
        self.tag = svi.tag+'_'+ self.id if svi.set_var else tmp_label
        self.set_namespace = svi.set_namespace
        self.parent_scope_id = svi.parent_scope_id
        self.set_var = tmp_label if svi.set_var else None
        self.set_var_ix = svi.set_var_ix if svi.set_var else None
        self.value = svi.value
        self.type=VariableType.PARAMETER
        self.path = svi.path

    def get_path_dot(self):
        return self.tag


def generate_equations(equations, equation_graph: Graph, scoped_equations, scope_variables, scope_ids, aliases):
    print('n var: ',len(scope_variables))

    scope_var_dot = [sv.get_path_dot() for sv in scope_variables.values()]
    #if not 'climatemachine.HX_element_1.HX_1_1.element_1_outside_1_1.t1.dp' in scope_var_dot:
    #    raise ValueError()

    #Replace individual assignments with a sum
    vars_assignments = {}
    vars_assignments_mappings = {}
    nodes_map = equation_graph.node_map

    logging.info('Remove simple assign chains')


    for n in nodes_map.values():
        if equation_graph.get(n, 'node_type') == NodeTypes.VAR:
            #Get target
            target = n
            target_edges_indcs, target_edges = equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val=['target', 'mapping'])

            for edge, edge_ix in zip(target_edges, target_edges_indcs):

                if not target in vars_assignments:
                    vars_assignments[target] = []
                    vars_assignments_mappings[target] = []

                vars_assignments[target].append(edge[0])
                vars_assignments_mappings[target].append(equation_graph.edges_attr['mappings'][edge_ix])
            if target in vars_assignments and len(vars_assignments[target])>1:
                for edge_ix in target_edges_indcs:
                    equation_graph.remove_edge(edge_ix)
                #Traverse value of assignment - might be  + + +
                #value_edge = equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val='value')[1][0]
                #value = value_edge[0]

                #visit_assign_value(vars_assignments[target], value, nodes_map, equation_graph)

                #equation_graph.remove_node(n)

    #equation_graph.as_graphviz('before eq', force=True)

    logging.info('create assignments')
    from tqdm import tqdm

    from copy import copy

    for ii, n in tqdm(enumerate(equation_graph.get_where_attr('node_type', NodeTypes.EQUATION))):

        #print(ii)
        for i, e in equation_graph.get_edges_for_node(start_node=n):
                #print(i)
                va = e[1].copy()
                if va in vars_assignments and len(vars_assignments[va])>1:

                    # Make new temp var
                    tmp_label = equation_graph.key_map[va] + '_tmp'
                    sv = equation_graph.get(e[1], 'scope_var')

                    #Create fake scope variables for tmp setvar

                    fake_sv = {}
                    svf=None
                    for i_, svi in tqdm(enumerate(scope_variables.values())):
                        if sv.set_var and svi.set_var == sv.set_var:

                            svf = TemporaryVar(svi, tmp_label)
                            #print('tmp_label: ',tmp_label)
                            #print('tmp_path: ',svf.get_path_dot())
                            fake_sv[d_u(svf.get_path_dot())]= svf

                    if not sv.set_var:
                        svf = TemporaryVar(sv, tmp_label)
                        fake_sv[d_u(svf.get_path_dot())] = svf

                    scope_variables.update(fake_sv)


                    tmp = equation_graph.add_node(key=tmp_label,  node_type=NodeTypes.TMP, name=tmp_label, ast=None, file='sum', label=tmp_label, ln=0,
                                 ast_type=None, scope_var=svf, ignore_existing=False)
                    # Add temp var to Equation target
                    #equation_graph.edges[i,1] = tmp

                    equation_graph.add_edge(n, tmp, e_type='target', arg_local=equation_graph.edges_attr['arg_local'][i[0]])
                    # Add temp var in var assignments
                    #print('sdfsdf: ',equation_graph.edges_attr['mappings'][i[0]])

                    vars_assignments_mappings[va][(nix:= vars_assignments[va].index(n))]= ':'
                    vars_assignments[va][nix] = tmp

        #vars_assignments.update(new_vars_assignments)
        #vars_assignments_mappings.update(new_vars_assignments_mappings)
                    #vars_assignments[va].append(tmp)
    #scope_variables.update(new_scope_variables)
    logging.info('Add mappings')
    for a, vals in vars_assignments.items():
        if len(vals)>1:
            ns = new_sum()
            nsn = equation_graph.add_node(key=ns, node_type=NodeTypes.SUM, name=ns, ast=None, file='sum', label=ns, ln=0, ast_type=None)
            equation_graph.add_edge(nsn, a, e_type='target')
            for v, mappings in zip(vals, vars_assignments_mappings[a]):
                equation_graph.add_edge(v, nsn, e_type='value', mappings=mappings)

    logging.info('Cleaning eq graph')
    equation_graph = equation_graph.clean()

    #equation_graph.as_graphviz('eq_sum_swap', force=True)
    llvm_funcs = {}
    mod_body = []
    #Loop over equation functions and generate code
    eq_vardefs={}
    logging.info('make equations for compilation')
    for eq_key, eq in equations.items():
        #print(eq)
        vardef = Vardef()
        #vardef__ = Vardef()
        vardef_llvm = Vardef_llvm()
        func, vardef_ = function_from_graph_generic(eq[2],eq_key.replace('.','_'), var_def_=vardef, decorators = ["njit"])
        #func__, vardef___ = function_from_graph_generic(eq[2], eq_key.replace('.', '_')+'_nojit', var_def_=vardef__,
         #                                           decorators=[])
        eq[2].lower_graph = None
        func_llvm, vardef__, signature, fname, args, targets = function_from_graph_generic_llvm(eq[2], eq_key.replace('.', '_'), var_def_=vardef_llvm)
        llvm_funcs[eq_key.replace('.', '_')]={'func_ast': func_llvm, 'signature': signature, 'name': fname, 'args': args, 'targets': targets}
        eq_vardefs[eq_key] = vardef

        mod_body.append(func)
        #mod_body.append(func__)

        mod_body.append(func_llvm)
    body=[]
    #Create a kernel of assignments and calls
    all_targeted = []
    all_read = []
    logging.info('Generate kernel')

    #equation_graph.as_graphviz('after')
    #equation_graph = equation_graph.clean()
    """
    equation_graph_clone = equation_graph.clone()

    for n in equation_graph_clone.node_map.values():
        s = equation_graph_clone.get_edges_for_node(start_node=n)
        e = equation_graph_clone.get_edges_for_node(end_node=n)
        if len(list(s))+len(list(e))<=1:
            equation_graph_clone.remove_node(n)
            #for i in set(ix_s +ix_e):
            #    equation_graph_clone.remove_edge(i)

    equation_graph_clone= equation_graph_clone.clean()
    """
    #equation_graph.as_graphviz('clean after', force=True)
    states = []
    deriv = []
    mapping = []
    other = []
    deriv_aliased = {}

    #vars_node_id = {sv.id: equation_graph.key_map[n] for n in
    #                equation_graph.get_where_attr('node_type', val=NodeTypes.VAR) if
    #                (sv := equation_graph.get(n, 'scope_var'))}
    vars_node_id = {}
    #scope_var_node = {equation_graph.key_map[n]: sv for n in
    #                  equation_graph.get_where_attr('node_type', val=[NodeTypes.VAR, NodeTypes.TMP]) if
    #                  (sv := equation_graph.get(n, 'scope_var'))}
    scope_var_node={}
    for sv_id, sv in scope_variables.items():
        full_tag = d_u(sv.get_path_dot())
        if not sv_id in vars_node_id:

            vars_node_id[sv_id] = full_tag



            if sv.type == VariableType.DERIVATIVE:
                if full_tag in aliases:
                    deriv_aliased[full_tag] = aliases[full_tag]

        if full_tag not in scope_var_node:
            scope_var_node[full_tag] = sv



        if sv.type == VariableType.STATE:

            states.append(vars_node_id[sv_id])
        elif sv.type == VariableType.DERIVATIVE:
            #print('deriv: ', sv.id)
            deriv.append(vars_node_id[sv_id])

    set_variables = {}
    for ix, sv in enumerate(scope_variables.values()):
        sv_tuple = (sv.id, sv, ix)
        #print(sv.id)
        #print('path: ',sv.get_path_dot())

        if sv.set_var:

            if not sv.set_var in set_variables:
                set_variables[sv.set_var] = [None] * sv.set_namespace.len_items

            set_variables[sv.set_var][sv.set_var_ix] = sv_tuple


    #for k, v in set_variables.items():
    #    print(k, ': ', v)

    topo_sorted_nodes = equation_graph.topological_nodes()
    body_def = []
    for n in topo_sorted_nodes:
        if (nt:= equation_graph.get(n, 'node_type')) == NodeTypes.EQUATION:
            #print('adding scope: ', eq_key)

            eq_key = scoped_equations[equation_graph.key_map[n]]
            eq = equations[eq_key]
            vardef = eq_vardefs[eq_key]

            a_indcs, a_edges = list(equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val='arg'))

            args_local = [equation_graph.key_map[ae[0]] for i, ae in zip(a_indcs, a_edges) if not equation_graph.edges_attr['arg_local'][i] == 'local']
            args_scope_var = [equation_graph.edges_attr['arg_local'][i] for i, ae in zip(a_indcs, a_edges) if not equation_graph.edges_attr['arg_local'][i]=='local']
            #all_read += args_local#[equation_graph.key_map[a] for a in args_local]

            t_indcs, t_edges = list(equation_graph.get_edges_for_node_filter(start_node=n, attr='e_type', val='target'))
            targets_local = [equation_graph.key_map[te[1]] for i, te in zip(t_indcs, t_edges) if not equation_graph.edges_attr['arg_local'][i] == 'local']
            targets_scope_var = [equation_graph.edges_attr['arg_local'][i] for i, ae in zip(t_indcs, t_edges) if not equation_graph.edges_attr['arg_local'][i]=='local']
            #all_targeted += targets_local#[equation_graph.key_map[tl] for tl in targets_local]# if equation_graph.nodes_map[tl][1].node_type != NodeTypes.TMP]
            ##for k, v in zip(args_scope_var+targets_scope_var, args_local + targets_local):
            #    print(k,': ',v)
            scope_vars = {'scope.'+k: v for k, v in zip(args_scope_var+targets_scope_var, args_local + targets_local)}

            args = [ast.Name(id=d_u(scope_vars[a])) for a in vardef.args]



            equation_graph.nodes_attr['scope_var'][n]= {'args': [scope_vars[a] for a in vardef.args], 'targets': [scope_vars[a] for a in vardef.targets]}
            if equation_graph.get(n, 'vectorized'):
                if len(vardef.targets) > 1:
                    targets = [ast.Tuple(
                        elts=[ast.Subscript(value=ast.Name(id=d_u(scope_vars[t])), slice=ast.Name(id='i')) for t in
                              vardef.targets])]
                else:
                    targets = [ast.Name(id=d_u(scope_vars[vardef.targets[0]]))]


                for t in vardef.targets:
                    body_def.append(ast.Assign(targets=[ast.Name(id=d_u(scope_vars[t]))], value=ast.Call(func=ast.Attribute(attr='empty', value=ast.Name(id='np')), args=[ast.Num(n=len(set_variables[scope_vars[t]]))], keywords=[])))

                for a in vardef.args:
                    _set_vars = [d_u(v[1].get_path_dot()) for v in set_variables[scope_vars[a]]]
                    all_read += _set_vars

                    body_def.append(ast.Assign(targets=[ast.Name(id=d_u(scope_vars[a]))], value=ast.List(elts=[ast.Name(id=set_v) for set_v in _set_vars])))



                body.append(

                    #For loop over items in set
                    ast.For(
                        body=[ast.Assign(targets=targets, value=ast.Call(func=ast.Name(id=scoped_equations[equation_graph.key_map[n]].replace('.','_')),
                                                                         args=[ast.Subscript(value=ast.Name(id=a), slice=ast.Index(value=ast.Name(id='i'))) for a in args], keywords=[]))],
                        orelse=[],
                        iter=ast.Call(func=ast.Name(id='range'), args=[ast.Num(n=len(set_variables[scope_vars[t]]))], keywords=[], target=ast.Name(id='i')),
                        target=ast.Name(id='i')
                    )
                )

                for t in vardef.targets:
                    #print(t)
                    #body.append(ast.Assign(
                    #    targets=[ast.Tuple(elts=[ast.Name(id=d_u(v[1].get_path_dot())) for v in set_variables[scope_vars[t]]])],
                    #                       value=ast.Name(id=d_u(scope_vars[t]))
                    #                       ))

                    if equation_graph.get(equation_graph.node_map[scope_vars[t]], "node_type") != NodeTypes.TMP:
                        all_targeted+=[d_u(v[1].get_path_dot()) for v in set_variables[scope_vars[t]]]
                        s=scope_vars[t]
                        setvar__=set_variables[scope_vars[t]]
                        b=2
                a = 1

            else:
                read_ = [d_u(scope_vars[a]) for a in vardef.args]

                all_read += read_
                if len(vardef.targets) > 1:
                    targets = [ast.Tuple(
                        elts=[ast.Name(id=d_u(scope_vars[t]))for t in
                              vardef.targets])]

                    all_targeted += [d_u(scope_vars[t]) for t in
                              vardef.targets]
                else:
                    targets = [ast.Name(id=d_u(scope_vars[vardef.targets[0]]))]
                    all_targeted.append(d_u(scope_vars[vardef.targets[0]]))

                body.append(ast.Assign(targets=targets, value=ast.Call(
                    func=ast.Name(id=scoped_equations[equation_graph.key_map[n]].replace('.', '_')), args=args,
                    keywords=[])))


        elif nt == NodeTypes.SUM:
            t_indcs, target_edges = list(equation_graph.get_edges_for_node_filter(start_node=n, attr='e_type', val='target'))
            v_indcs, value_edges = list(equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val='value'))
            t = target_edges[0][1]
            #all_targeted.append(equation_graph.key_map[(])


            if (t_sv:= equation_graph.get(t, 'scope_var')).set_var:
                l_mapping = len(set_variables[t_sv.set_var])
                mappings = {':': [], 'ix': []}
                #make a list of assignments to each index in t
                for v_ix, v in zip(v_indcs, value_edges):
                    if (nt:=equation_graph.get(v[0], 'node_type')) == NodeTypes.VAR or nt == NodeTypes.TMP:

                        if (mix:=equation_graph.edges_attr['mappings'][v_ix]) == ':':
                            mappings[':'].append(d_u(equation_graph.key_map[v[0]]))

                        elif isinstance(mix, list):
                            sums = {}
                            for m in mix:
                                if not m[1] in sums:
                                    sums[m[1]] = []
                                sums[m[1]].append(m[0])
                            mappings['ix'].append((d_u(equation_graph.key_map[v[0]]), sums))

                        else:
                            raise ValueError(f'mapping indices not specified!{equation_graph.edges_attr["mappings"][v_ix]}, {equation_graph.key_map[t]} <- {equation_graph.key_map[v[0]]}')


                          #body.append(
                          #      ast.Assign(targets=[ast.Name(id=t_sv.set_var)],

                          #                 value=ast.Name(id=d_u(equation_graph.key_map[v[0]])))
                          #  )

                    else:
                        raise ValueError(f'this must be a mistake {equation_graph.key_map[v[0]]}')

                if len(mappings[':'])>0:
                    if len(mappings[':'])>1:
                        prev = None
                    else:
                        prev = ast.Num(n=0)
                    for mcolon in mappings[':']:

                        if prev:
                            #print('prev: ',prev)
                            prev = ast.BinOp(left=prev, right=ast.Name(id=mcolon), op=ast.Add())
                        else:
                            prev = ast.Name(id=mcolon)

                    body.append(ast.Assign(targets=[ast.Name(id=d_u(t_sv.set_var))], value = prev))
                else:
                    body.append(ast.Assign(targets=[ast.Name(id=t)], value=ast.Call(func=ast.Attribute(attr='empty', value=ast.Name(id='np')), args=[ast.Num(n=l_mapping)], keywords=[])))
                    pass
                for m_ix in mappings['ix']:

                    #for m in m_ix[1].values():
                    m_ix1_keys = list(m_ix[1].keys())
                    tar = ast.Subscript(
                        slice=ast.Index(value=ast.Tuple(elts=[ast.Num(n=mmix) for mmix in m_ix1_keys]) if len(m_ix1_keys)>1 else ast.Num(n=m_ix1_keys[0])),
                        value=ast.Name(id=d_u(t_sv.set_var)))

                    def add_ast_gen(elts_to_sum, op=ast.Add()):
                        prev = None
                        for ets in elts_to_sum:
                            if prev:
                                prev = ast.BinOp(op=op, left=prev, right=ets)
                            else:
                                prev = ets
                        return prev
                    var_elts = [
                        add_ast_gen([ast.Name(id=d_u(m_ix[0])) if m__ is None else ast.Subscript(slice=ast.Index(ast.Num(n=m__)), value=ast.Name(id=d_u(m_ix[0]))) for m__ in m_]) for m_ in m_ix[1].values()]

                    [[all_read.append(d_u(m_ix[0])) for m__ in m_ if m__ is None] for m_ in m_ix[1].values()]

                    var = ast.Tuple(
                        elts=var_elts
                    ) if len(var_elts)>1 else var_elts[0]

                    body.append(ast.AugAssign(target=tar, value=var, op=ast.Add()))


            else:
                all_targeted.append(d_u(equation_graph.key_map[t]))

                values = []
                for v in value_edges:
                    if equation_graph.get(v[0], 'node_type') == NodeTypes.VAR:
                         all_read.append(d_u(equation_graph.key_map[v[0]]))
                    values.append(ast.Name(id=d_u(equation_graph.key_map[v[0]])))

                if len(values)>1:
                    prev = None
                    for v in values:
                        if prev:
                            prev = ast.BinOp(op=ast.Add(), left=v, right=prev)
                        else:
                            prev=v
                    assign = ast.Assign(targets=[ast.Name(id=d_u(equation_graph.key_map[target_edges[0][1]]))], value=prev)
                else:
                    assign = ast.Assign(targets=[ast.Name(id=d_u(equation_graph.key_map[target_edges[0][1]]))],
                                        value=values[0])

                body.append(assign)

        elif nt == NodeTypes.VAR or nt == NodeTypes.TMP:
            pass

        else:
            raise ValueError('Unused node: ', equation_graph.key_map[n])

    #for a in aliases.values():
    #    if not a in all_targeted:
    #        all_read.append(d_u(a))

    def non_unique_check(listname_, list_):

        if len(list_) > len(set(list_)):
            import collections
            raise ValueError(
                f'Non unique {listname_}: {[item for item, count in collections.Counter(list_).items() if count > 1]}')

    def contains_dot(str_list):
        for s in str_list:
            if '.' in s:
                raise ValueError(f'. in {s}')

    for sv_id, sv in scope_variables.items():
        full_tag = d_u(sv.get_path_dot())
        if not sv_id in vars_node_id:

            vars_node_id[sv_id] = full_tag



            if sv.type == VariableType.DERIVATIVE:
                if full_tag in aliases:
                    deriv_aliased[full_tag] = aliases[full_tag]

        if full_tag not in scope_var_node:
            scope_var_node[full_tag] = sv

    all_read = set(all_read)
    contains_dot(all_read)
    all_targeted = set(all_targeted)
    contains_dot(all_targeted)
    all_must_init = set(all_read).difference(all_targeted)

    lenstates = len(states)
    lenderiv = len(deriv)

    vars_init = [d_u(s) for s in states.copy()]

    vars_init += list(all_must_init.difference(vars_init))

    leninit = len(vars_init)

    non_unique_check('deriv: ', deriv)
    non_unique_check('all targeted: ', all_targeted)

    vars_update = [d_u(d) for d in deriv.copy()]
    vars_update += [at for at in all_targeted if at not in vars_update]
    #vars_update += list(set(all_targeted).difference(vars_update))



    for s, d in zip(states, deriv):
        if not d[:-4] == s:
            #print(d, ' ', s)
            raise IndexError('unsorted derivs')


    indcs = (lenstates, leninit, lenderiv)

    variables = vars_init + vars_update

    variables += set([d_u(sv.get_path_dot()) for sv in scope_variables.values()]).difference(variables)
    variables_dot = [scope_var_node[v].get_path_dot() for v in variables]
    #if not 'climatemachine.HX_element_1.HX_1_1.element_1_outside_1_1.t1.dp' in variables_dot:
    #    raise ValueError()

    non_unique_check('initialized vars', vars_init)
    len_vars_init_ = len(vars_init)
    vars_init += deriv
    non_unique_check('updated vars', vars_update)

    non_unique_check('variables', variables)

    #if len(vars_update) > len(set(vars_update)):
      #  raise ValueError('Non unique update vars')




    body = [
        ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id=d_u(i)) for i in vars_init[len(states):]])], value=ast.Subscript(slice=ast.Slice(lower=ast.Num(n=len(states)), upper=ast.Num(n=len(vars_init)), step=None), value=ast.Name(id='variables'))),
        ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id=d_u(s)) for s in states])], value=ast.Name(id='y')),
           ] + body_def + body

    llvm_sequence = []
    #llvm_sequence = [{'func': 'load', 'ix': ix, 'var': vi, 'arg': 'variables'} for vi, ix  in zip(vars_init[len(states):], range(len(states), len(vars_init)))]
    llvm_sequence += [{'func': 'load', 'ix': ix+lenstates, 'var': v, 'arg': 'variables'} for ix, v in enumerate(vars_init[lenstates:])]
    llvm_sequence += [{'func': 'load', 'ix': ix, 'var': s, 'arg': 'y'} for ix, s in enumerate(states)]
    llvm_end_seq = []
    #llvm_end_seq = [{'func': 'store', 'arg': 'variables', 'ix': ix, 'var': u} for u, ix in zip(vars_update, range(len(vars_init), len(vars_init)+len(vars_update)))]
    llvm_end_seq += [{'func': 'store', 'arg': 'variables', 'ix': ix, 'var': u} for u, ix in zip(states, range(0, lenstates))]
    #llvm_end_seq += [{'func': 'store', 'arg': 'variables', 'ix': ix, 'var': d} for ix, d in enumerate(deriv)]


    [body.append(ast.Assign(targets=[ast.Name(id=d_u(d))], value = ast.Name(id=d_u(a)))) for d, a in deriv_aliased.items()]

    # Add code for updating variables

    elts_vu = [(ast.Subscript(value=ast.Name(id=d_u(svn_.set_var)), slice=ast.Index(ast.Num(n=svn_.set_var_ix))) if (svn_:=scope_var_node[u]).set_var else ast.Name(id=d_u(u))) for i, u in enumerate(vars_update)]

    body.append(ast.Assign(targets=[ast.Subscript(slice=ast.Slice(lower=ast.Num(n=len_vars_init_), upper=ast.Num(n=len_vars_init_+len(vars_update)), step=None),
                                           value=ast.Name(id='variables'))],
                           value=ast.Tuple(elts=elts_vu)))

    # Add code for updating derivatives
    body.append(ast.Assign(value=ast.Tuple(elts=[
        ast.Subscript(value=ast.Name(id=d_u(svn_.set_var)), slice=ast.Index(ast.Num(n=svn_.set_var_ix))) if (svn_:=scope_var_node[u]).set_var else ast.Name(id=d_u(u)) for u  in states
    ]), targets=[ast.Subscript(slice=ast.Slice(lower=ast.Num(n=0), upper=ast.Num(n=len(states)), step=None), value=ast.Name(id='variables'))]))

    #body.append(ast.Return(value=ast.Call(func=ast.Name(id='np.array'), args=[ast.List(elts=[ast.Name(id=d) for d in deriv]), ast.Name(id='np.float64')], keywords=[])))
    body.append(ast.Return(value=ast.Subscript(slice=ast.Slice(lower=ast.Num(n=leninit), upper=ast.Num(n=leninit+lenderiv), step=None), value=ast.Name(id='variables'))))
    kernel_args = dot_dict(args=[ast.Name(id='variables'), ast.Name(id='y')], vararg=None, defaults=[], kwarg=None)

    #self.variables[var.id].path.path[self.system.id]


    skip_kernel = False
    if not skip_kernel:
        #mod_body.append(wrap_function('kernel', body, decorators=["njit('float64[:](float64[:],float64[:])')"], args=kernel_args))
        mod_body.append(
        wrap_function('kernel_nojit', body, decorators=[], args=kernel_args))
    logging.info('generate program')
    #run_program_source, lib_body, program, indices, llvm_program = generate_program(equation_graph, variables, indcs, deriv_aliased)
    logging.info('done program')
    #mod_body+=lib_body

    #LLVM
    #llvm_sequence += llvm_program + llvm_end_seq
    #llvm_sequence += llvm_end_seq


    source = generate_code_file(mod_body, 'kernel.py')
    logging.info('compiling...')



    import timeit
    print('Compile time: ', timeit.timeit(
        lambda: exec('from kernel import *', globals()), number=1))

    #print(scope_var_node.keys())
    #variables_values = np.array([scope_var_node[v].value for v in variables], dtype=np.float64)
    variables_values = np.array([scope_var_node[v].value for v in variables], dtype=np.float64)

    #variables_values_ = np.array([scope_var_node[v].value for v in variables], dtype=np.float64)

    #for v, vv in zip(variables, variables_values_):
    #    print(v,': ',vv)
#    asfsdf=sdfsdf
    from numerous.engine.model.generate_llvm import generate as generate_llvm

    #for fn, f in llvm_funcs.items():
    #    f['func'] = globals()[f['name']]

    #for l in llvm_sequence:
    #    if 'ext_func' in l:
    #        l['ext_func'] = llvm_funcs[l['ext_func']]['name']


    #from numba import njit, float64, int64
    #logging.info('generate llvm')
    #diff_llvm, var_func, var_func_set, max_deriv = generate_llvm(llvm_sequence, llvm_funcs.values(), variables, variables_values, leninit, lenderiv)

    ###TESTS####
    y = variables_values[:lenderiv]
    y_ = variables_values[:lenderiv].astype(np.float64)
    #variables_ = variables_values.astype(np.float32)#np.array([0, 1, 0, 1, 0, 1, 0, 1], np.float32)

    from time import time
    N = 10000

    #@njit('float64[:](float64[:], int64)')
    #def diff_bench_llvm(y, N):

    #    for i in range(N):
    #        derivatives = diff_llvm(y)

    #    return derivatives

    tic = time()
    #derivs_llvm = diff_bench_llvm(y, N)
    toc = time()
    #llvm_vars = var_func(0)
    #print('llvm derivs: ', list(zip(deriv, derivs_llvm)))
    #print('llvm vars: ', list(zip(variables, var_func(0))))
    #print(f'Exe time llvm - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)


    N = 5
    if not skip_kernel:
        def test_kernel_nojit(variables, y):
            for i in range(N):
                deriv = kernel_nojit(variables, y)
                print(deriv)
            return deriv
        #print(y)
        tic = time()
        deriv_no_jot = test_kernel_nojit(variables_values, y)
        toc = time()

        #sdfsdf=sdfsdfsdf
        class AssemlbedModel():
            def __init__(self, vars, vals):
                self.variables = vars
                self.init_vals = vals

                @njit
                def diff(y):
                    with objmode(derivs='float64[:]'):  # annotate return type
                        # this region is executed by object-mode.'
                        #print(y)
                        derivs = self.diff__(y)
                        #print(derivs)
                    return derivs.copy()

                self.diff = diff

                @njit
                def var_func(i):
                    with objmode(vrs='float64[:]'):  # annotate return type
                        # this region is executed by object-mode.
                        vrs = self.vars__()
                        #print(vrs)
                    return vrs.copy()

                self.var_func = var_func

            def diff__(self, y):
                return kernel_nojit(self.init_vals, y)

            def vars__(self):
                #for v, vv in zip(self.variables, self.init_vals):
                #    print(v,': ',vv)
                return self.init_vals

        am = AssemlbedModel(variables, variables_values)
        diff_ = am.diff
        var_func_ = am.var_func


        #print(deriv_no_jot)
        print(f'Exe time flat no jit - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)

        print('no jit derivs: ', list(zip(deriv, deriv_no_jot)))
        print('no jit vars: ', list(zip(variables, am.var_func(0))))

        print('var diff')
        #for k, v_n, v_llvm in zip(variables, am.var_func(0), var_func(0)):
        #    print(k,': ',v_n,' ',v_llvm,' diff: ', v_n-v_llvm)

        print('deriv diff')
        #for k, v_n, v_llvm in zip(deriv, deriv_no_jot, derivs_llvm):
        #    if abs(v_n) >1e-20:
        #        rel_diff = (v_n - v_llvm) / v_n
        #    else:
        #        rel_diff = 0

        #    print(k,': ',v_n,' ',v_llvm,' rel diff: ', rel_diff)
        #    if rel_diff>0.001:
        #        raise ValueError(f'Arg {k}, {v_n}, {v_llvm}, {rel_diff}')

        #print('Exe time kernel nojit timeit: ', timeit.timeit(
        #    lambda: kernel_nojit(variables_values, y), number=N) / N)

        N = 10000

        #@njit('void(float64[:], float64[:])')
        #def test_kernel(variables, y):
        #    for i in range(N):
        #        kernel(variables, y)

        print('First kernel call results: ')
        #print(kernel(variables_values, y))

        tic = time()
        #test_kernel(variables_values, y)
        toc = time()

        print(f'Exe time flat - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)
        #print('Exe time kernel timeit: ', timeit.timeit(
        #    lambda: kernel(variables_values, y), number=N) / N)
    count = 0
    #for v, v_llvm, v_kernel in zip(variables, llvm_vars, variables_values):
    #    err = abs(v_llvm - v_kernel)/abs(v_llvm + v_kernel)*2
    #    if err>1e-3:
    #        print(v,': ',v_llvm,' ',v_kernel, ' ', err)
        #count+=1
        #print(count)

    #sdfsdf=sdfsdf
    ###TEST PROGRAM
    #spec = [
    #    ('program', int64[:, :]),
    #    ('indices', int64[:]),


    #]
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
    #print(dp.diff(variables_values, y))

    #tic = time()
    #dp.test(variables_values, y)
    #toc = time()



    #print(f'Exe time program - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)

    #print('Exe time program timeit: ', timeit.timeit(
#        lambda: dp.diff(variables_values, y), number=N)/N)
    llvm_ = False
    if llvm_:
        return diff_llvm, var_func, variables_values, variables_dot, scope_var_node
    else:
        return diff_, var_func_, variables_values, variables_dot, scope_var_node