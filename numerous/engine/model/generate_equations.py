from numerous.engine.model.graph import Graph
from numerous.engine.model.utils import NodeTypes, recurse_Attribute, wrap_function, dot_dict, generate_code_file
from numerous.engine.model.parser_ast import attr_ast, function_from_graph_generic, function_from_graph_generic_llvm#, EquationNode, EquationEdge
from numerous.engine.variables import VariableType
from numerous.engine.model.generate_program import generate_program
import logging
import ast
from numba import njit
import numpy as np

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

        ix, left_edges = equation_graph.get_edges_for_node_filter(end_node=value, attr='e_type', val='left')

        left_edge = left_edges[0]
        left = left_edge[0]

        visit_assign_value(target, left, nodes_map, equation_graph)
        equation_graph.mark_remove_edge(left_edge)

        ix, right_edges = equation_graph.get_edges_for_node_filter(end_node=value, attr='e_type', val='right')
        right_edge = right_edges[0]

        right = right_edge[0]

        visit_assign_value(target, right, nodes_map, equation_graph)
        equation_graph.mark_remove_edge(right_edge)

        equation_graph.mark_remove_node(value[0])

    else:

        target.append(value)



def generate_equations(equations, equation_graph: Graph, scoped_equations, scope_variables):
    #Replace individual assignments with a sum
    vars_assignments = {}
    nodes_map = equation_graph.node_map

    logging.info('Remove simple assign chains')

    for n in nodes_map.values():
        if equation_graph.get(n, 'node_type') == NodeTypes.ASSIGN:
            #Get target
            target_edge = equation_graph.get_edges_for_node_filter(start_node=n, attr='e_type', val='target')[1][0]
            target = target_edge[1]

            if not target in vars_assignments:
                vars_assignments[target] = []

            #Traverse value of assignment - might be  + + +
            value_edge = equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val='value')[1][0]
            value = value_edge[0]

            visit_assign_value(vars_assignments[target], value, nodes_map, equation_graph)

            equation_graph.remove_node(n)

    #equation_graph.as_graphviz('before eq', force=True)

    logging.info('create assignments')

    for n in equation_graph.get_where_attr('node_type', NodeTypes.EQUATION):

        for i, e in equation_graph.get_edges_for_node(start_node=n):
                va = e[1].copy()
                if va in vars_assignments:

                    # Make new temp var
                    tmp_label = equation_graph.key_map[va] + '_tmp'
                    tmp = equation_graph.add_node(key=tmp_label,  node_type=NodeTypes.TMP, name=tmp_label, ast=None, file='sum', label=tmp_label, ln=0,
                                 ast_type=None, scope_var=equation_graph.get(e[1], 'scope_var'), ignore_existing=False)
                    # Add temp var to Equation target
                    equation_graph.edges[i,1] = tmp

                    # Add temp var in var assignments

                    vars_assignments[va].append(tmp)

    for a, vals in vars_assignments.items():
        if len(vals)>1:
            ns = new_sum()
            nsn = equation_graph.add_node(key=ns, node_type=NodeTypes.SUM, name=ns, ast=None, file='sum', label=ns, ln=0, ast_type=None)
            equation_graph.add_edge(nsn,a, e_type='target')
            for v in vals:

                equation_graph.add_edge(v, nsn, e_type='value')

    llvm_funcs = {}
    mod_body = []
    #Loop over equation functions and generate code
    eq_vardefs={}
    logging.info('make equations for compilation')
    for eq_key, eq in equations.items():
        vardef = Vardef()
        vardef_llvm = Vardef_llvm()
        func, vardef_ = function_from_graph_generic(eq[2],eq_key.replace('.','_'), var_def_=vardef, decorators = ["njit"])
        eq[2].lower_graph = None
        func_llvm, vardef__, signature, fname, args, targets = function_from_graph_generic_llvm(eq[2], eq_key.replace('.', '_'), var_def_=vardef_llvm)
        llvm_funcs[eq_key.replace('.', '_')]={'func_ast': func_llvm, 'signature': signature, 'name': fname, 'args': args, 'targets': targets}
        eq_vardefs[eq_key] = vardef

        mod_body.append(func)
        mod_body.append(func_llvm)
    body=[]
    #Create a kernel of assignments and calls
    all_targeted = []
    all_read = []
    logging.info('Generate kernel')

    equation_graph.as_graphviz('after')
    equation_graph = equation_graph.clean()

    equation_graph.as_graphviz('clean after', force=True)
    topo_sorted_nodes = equation_graph.topological_nodes()

    #sdfsdf=sdfsdffdf
    for n in topo_sorted_nodes:
        if (nt:= equation_graph.get(n, 'node_type')) == NodeTypes.EQUATION:
            #print('adding scope: ', eq_key)

            eq_key = scoped_equations[equation_graph.key_map[n]]
            eq = equations[eq_key]
            vardef = eq_vardefs[eq_key]

            a_indcs, a_edges = list(equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val='arg'))

            args_local = [equation_graph.key_map[ae[0]] for i, ae in zip(a_indcs, a_edges) if not equation_graph.edges_attr['arg_local'][i] == 'local']
            args_scope_var = [equation_graph.edges_attr['arg_local'][i] for i, ae in zip(a_indcs, a_edges) if not equation_graph.edges_attr['arg_local'][i]=='local']
            all_read += args_local#[equation_graph.key_map[a] for a in args_local]

            t_indcs, t_edges = list(equation_graph.get_edges_for_node_filter(start_node=n, attr='e_type', val='target'))
            targets_local = [equation_graph.key_map[te[1]] for i, te in zip(t_indcs, t_edges) if not equation_graph.edges_attr['arg_local'][i] == 'local']
            targets_scope_var = [equation_graph.edges_attr['arg_local'][i] for i, ae in zip(t_indcs, t_edges) if not equation_graph.edges_attr['arg_local'][i]=='local']
            all_targeted += targets_local#[equation_graph.key_map[tl] for tl in targets_local]# if equation_graph.nodes_map[tl][1].node_type != NodeTypes.TMP]

            scope_vars = {'scope.'+k: v for k, v in zip(args_scope_var+targets_scope_var, args_local + targets_local)}

            args = [ast.Name(id=scope_vars[a]) for a in vardef.args]

            if len(vardef.targets)>1:
                targets = [ast.Tuple(elts=[ast.Name(id=scope_vars[t]) for t in vardef.targets])]
            else:
                targets = [ast.Name(id=scope_vars[vardef.targets[0]])]

            equation_graph.nodes_attr['scope_var'][n]= {'args': [scope_vars[a] for a in vardef.args], 'targets': [scope_vars[a] for a in vardef.targets]}

            body.append(ast.Assign(targets=targets, value=ast.Call(func=ast.Name(id=scoped_equations[equation_graph.key_map[n]].replace('.','_')), args=args, keywords=[])))

        if nt == NodeTypes.SUM:
            t_indcs, target_edges = list(equation_graph.get_edges_for_node_filter(start_node=n, attr='e_type', val='target'))
            v_indcs, value_edges = list(equation_graph.get_edges_for_node_filter(end_node=n, attr='e_type', val='value'))

            all_targeted.append(equation_graph.key_map[target_edges[0][1]])
            values = []
            for v in value_edges:
                if equation_graph.get(v[0], 'node_type') == NodeTypes.VAR:
                    all_read.append(equation_graph.key_map[v[0]])
                values.append(ast.Name(id=equation_graph.key_map[v[0]]))

            if len(values)>1:
                prev = None
                for v in values:
                    if prev:
                        prev = ast.BinOp(op=ast.Add(), left=v, right=prev)
                    else:
                        prev=v
                assign = ast.Assign(targets=[ast.Name(id=equation_graph.key_map[target_edges[0][1]])], value=prev)
            else:
                assign = ast.Assign(targets=[ast.Name(id=equation_graph.key_map[target_edges[0][1]])],
                                    value=values[0])

            body.append(assign)

    all_must_init = set(all_read).difference(all_targeted)

    vars_node_id = {sv.id: equation_graph.key_map[n] for n in equation_graph.get_where_attr('node_type', val=NodeTypes.VAR) if (sv:=equation_graph.get(n,'scope_var'))}

    scope_var_node = {equation_graph.key_map[n]: sv for n in equation_graph.get_where_attr('node_type', val=[NodeTypes.VAR, NodeTypes.TMP]) if (sv:=equation_graph.get(n,'scope_var'))}

    states = []
    deriv = []
    mapping = []
    other = []
    for sv_id, sv in scope_variables.items():

        if sv.type == VariableType.STATE:
            states.append(vars_node_id[sv.id])
        elif sv.type == VariableType.DERIVATIVE:
            deriv.append(vars_node_id[sv.id])

    vars_init = states.copy()
    lenstates=len(vars_init)
    vars_init += list(all_must_init.difference(vars_init))
    leninit=  len(vars_init)
    vars_update = deriv.copy()
    lenderiv = len(vars_update)
    vars_update += list(set(all_targeted).difference(vars_update))
    indcs = (lenstates, leninit, lenderiv)
    variables = vars_init + vars_update

    body = [
        ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id=i) for i in vars_init[len(states):]])], value=ast.Subscript(slice=ast.Slice(lower=ast.Num(n=len(states)), upper=ast.Num(n=len(vars_init)), step=None), value=ast.Name(id='variables'))),
        ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id=s) for s in states])], value=ast.Name(id='y')),
           ] + body

    llvm_sequence = [{'func': 'load', 'ix': ix, 'var': vi, 'arg': 'variables'} for vi, ix  in zip(vars_init[len(states):], range(len(states), len(vars_init)))]

    llvm_sequence += [{'func': 'load', 'ix': ix, 'var': s, 'arg': 'y'} for ix, s in enumerate(states)]

    llvm_end_seq = [{'func': 'store', 'arg': 'variables', 'ix': ix, 'var': u} for u, ix in zip(vars_update, range(len(vars_init), len(vars_init)+len(vars_update)))]

    llvm_end_seq += [{'func': 'store', 'arg': 'deriv', 'ix': ix, 'var': d} for ix, d in enumerate(deriv)]


    body.append(ast.Assign(targets=[ast.Subscript(slice=ast.Slice(lower=ast.Num(n=len(vars_init)), upper=ast.Num(n=len(vars_init)+len(vars_update)), step=None),
                                           value=ast.Name(id='variables'))],
                           value=ast.Tuple(elts=[ast.Name(id=u) for u in vars_update])))

    body.append(ast.Return(value=ast.Call(func=ast.Name(id='np.array'), args=[ast.List(elts=[ast.Name(id=d) for d in deriv]), ast.Name(id='np.float64')], keywords=[])))

    kernel_args = dot_dict(args=[ast.Name(id='variables'), ast.Name(id='y')], vararg=None, defaults=[], kwarg=None)

    skip_kernel = False
    if not skip_kernel:
        mod_body.append(wrap_function('kernel', body, decorators=["njit('float64[:](float64[:],float64[:])')"], args=kernel_args))
        mod_body.append(
        wrap_function('kernel_nojit', body, decorators=[], args=kernel_args))
    logging.info('generate program')
    run_program_source, lib_body, program, indices, llvm_program = generate_program(equation_graph, variables, indcs)
    mod_body+=lib_body

    #LLVM
    llvm_sequence += llvm_program + llvm_end_seq

    source = generate_code_file(mod_body, 'kernel.py')
    logging.info('compiling...')



    import timeit
    print('Compile time: ', timeit.timeit(
        lambda: exec(source, globals()), number=1))

    variables_values = np.array([scope_var_node[v].value for v in variables], dtype=np.float64)
    variables_values_ = np.array([scope_var_node[v].value for v in variables], dtype=np.float32)
    from numerous.engine.model.generate_llvm import generate as generate_llvm

    for fn, f in llvm_funcs.items():
        f['func'] = globals()[f['name']]

    for l in llvm_sequence:
        if 'ext_func' in l:
            l['ext_func'] = llvm_funcs[l['ext_func']]['name']


    from numba import njit, float64, int64

    diff_llvm, var_func, var_func_set, max_deriv = generate_llvm(llvm_sequence, llvm_funcs.values(), variables, variables_values_, leninit, lenderiv)

    ###TESTS####
    y = variables_values[:lenderiv]
    y_ = variables_values[:lenderiv].astype(np.float32)
    #variables_ = variables_values.astype(np.float32)#np.array([0, 1, 0, 1, 0, 1, 0, 1], np.float32)

    from time import time
    N = 10000

    @njit('float32[:](float32[:], int32)')
    def diff_bench_llvm(y, N):

        for i in range(N):
            derivatives = diff_llvm(y)

        return derivatives

    tic = time()
    derivs = diff_bench_llvm(y_, N)
    toc = time()
    print('llvm derivs: ', list(zip(deriv, derivs)))
    print(f'Exe time llvm - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)


    N = 1000
    if not skip_kernel:
        def test_kernel_nojit(variables, y):
            for i in range(N):
                kernel_nojit(variables, y)



        tic = time()
        test_kernel_nojit(variables_values, y)
        toc = time()

        print(f'Exe time flat no jit - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)
        print('Exe time kernel nojit timeit: ', timeit.timeit(
            lambda: kernel_nojit(variables_values, y), number=N) / N)

        N = 10000

        @njit('void(float64[:], float64[:])')
        def test_kernel(variables, y):
            for i in range(N):
                kernel(variables, y)

        print('First kernel call results: ')
        #print(kernel(variables_values, y))

        tic = time()
        test_kernel(variables_values, y)
        toc = time()

        print(f'Exe time flat - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)
        print('Exe time kernel timeit: ', timeit.timeit(
            lambda: kernel(variables_values, y), number=N) / N)

    ###TEST PROGRAM
    spec = [
        ('program', int64[:, :]),
        ('indices', int64[:]),


    ]
    N = 10000
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

    dp = DiffProgram(np.array(program, np.int64), np.array(indices, np.int64))


    print('First prgram call results: ')
    #print(dp.diff(variables_values, y))

    tic = time()
    dp.test(variables_values, y)
    toc = time()



    print(f'Exe time program - {N} runs: ', toc - tic, ' average: ', (toc - tic) / N)

    print('Exe time program timeit: ', timeit.timeit(
        lambda: dp.diff(variables_values, y), number=N)/N)



    return diff_llvm, var_func, variables_values, variables