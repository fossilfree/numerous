from numerous.engine.model.graph import Graph
from numerous.engine.model.utils import NodeTypes, recurse_Attribute, wrap_function, dot_dict, generate_code_file
from numerous.engine.model.parser_ast import attr_ast, function_from_graph_generic, EquationNode, EquationEdge
from numerous.engine.variables import VariableType
from numerous.engine.model.generate_program import generate_program

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

        #ix = self.vars_inds_map.index(var)

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
    if value[1].node_type == NodeTypes.OP:
        left_edge = equation_graph.edges_end(value, 'left')[0]
        left = nodes_map[left_edge[0]]

        visit_assign_value(target, left, nodes_map, equation_graph)

        right_edge = equation_graph.edges_end(value, 'right')[0]
        right = nodes_map[right_edge[0]]

        visit_assign_value(target, right, nodes_map, equation_graph)

        equation_graph.remove_node(value[0])
        equation_graph.remove_edge(left_edge)
        equation_graph.remove_edge(right_edge)

    else:

        target.append(value)



def generate_equations(equations, equation_graph: Graph, scoped_equations, scope_variables):
    #Replace individual assignments with a sum
    vars_assignments = {}
    nodes_map = equation_graph.nodes_map
    print(equation_graph.edges)
    for n in equation_graph.get_nodes():
        if n[1].node_type == NodeTypes.ASSIGN:
            #Get target
            target_edge = equation_graph.edges_start(n, 'target')[0]
            target = nodes_map[target_edge[1]]

            if not target[0] in vars_assignments:
                vars_assignments[target[0]] = []

            #Traverse value of assignment - might be  + + +
            value_edge = equation_graph.edges_end(n, 'value')[0]
            value = nodes_map[value_edge[0]]

            visit_assign_value(vars_assignments[target[0]], value, nodes_map, equation_graph)

            equation_graph.remove_edge(value_edge)
            equation_graph.remove_edge(target_edge)
            equation_graph.remove_node(n[0])

    for i, e in enumerate(equation_graph.edges.copy()):
        if nodes_map[e[0]][1].node_type == NodeTypes.EQUATION:
            if e[1] in vars_assignments:

                # Make new temp var
                tmp_label = e[1] + '_tmp'
                tmp = EquationNode(id=tmp_label, node_type=NodeTypes.TMP, name=tmp_label, ast=None, file='sum', label=tmp_label, ln=0,
                             ast_type=None, scope_var=nodes_map[e[1]][1].scope_var)
                # Add temp var to Equation target
                equation_graph.add_node((tmp_label, tmp, tmp_label))
                equation_graph.edges[i] = (e[0], tmp_label, e[2])

                # Add temp var in var assignments

                vars_assignments[e[1]].append(nodes_map[tmp_label])



    print(vars_assignments)

    for a, vals in vars_assignments.items():
        ns = new_sum()
        equation_graph.add_node((ns, EquationNode(id=ns, node_type=NodeTypes.SUM, name=ns, ast=None, file='sum', label=ns, ln=0, ast_type=None), ns))
        equation_graph.add_edge((ns, a, EquationEdge(label='target', start=ns, end=a)))
        for v in vals:
            equation_graph.add_edge((v[0], ns, EquationEdge(label='value', start=v[0], end=ns)))

    #equation_graph.as_graphviz('sum')

    mod_body = []
    #Loop over equation functions and generate code
    eq_vardefs={}
    for eq_key, eq in equations.items():
        vardef = Vardef()
        func, vardef_ = function_from_graph_generic(eq[2],eq_key.replace('.','_'), var_def_=vardef, decorators = ["njit"])
        print('args: ', vardef.args)
        print('targs: ', vardef.targets)
        eq_vardefs[eq_key] = vardef

        mod_body.append(func)
    body=[]
    #Create a kernel of assignments and calls
    #print(equation_graph.edges)
    all_targeted = []
    all_read = []
    for n in equation_graph.topological_nodes():
        if n[1].node_type == NodeTypes.EQUATION:
            eq_key = scoped_equations[n[0]]
            eq = equations[eq_key]
            vardef = eq_vardefs[eq_key]

            args_local = [ae[0] for ae in equation_graph.edges_end(n, 'args') if equation_graph.nodes_map[ae[0]][1].scope_var]
            all_read += args_local
            targets_local = [te[1] for te in equation_graph.edges_start(n, 'target') if equation_graph.nodes_map[te[1]][1].scope_var]
            all_targeted += [tl for tl in targets_local if equation_graph.nodes_map[tl][1].node_type != NodeTypes.TMP]
            scope_vars = {'scope.'+equation_graph.nodes_map[al][1].scope_var.tag: al for al in args_local + targets_local}
            print(scope_vars)


            args = [ast.Name(id=scope_vars[a]) for a in vardef.args]

            if len(vardef.targets)>1:
                targets = [ast.Tuple(elts=[ast.Name(id=scope_vars[t]) for t in vardef.targets])]
            else:
                targets = [ast.Name(id=scope_vars[vardef.targets[0]])]
            body.append(ast.Assign(targets=targets, value=ast.Call(func=ast.Name(id=scoped_equations[n[0]].replace('.','_')), args=args, keywords=[])))

        if n[1].node_type == NodeTypes.SUM:
            target_edges = equation_graph.edges_start(n, 'target')
            value_edges = equation_graph.edges_end(n, 'value')
            all_targeted.append(target_edges[0][1])
            values = []
            for v in value_edges:

                if nodes_map[v[0]][1].node_type == NodeTypes.VAR or nodes_map[v[0]][1].node_type == NodeTypes.DERIV or nodes_map[v[0]][1].node_type == NodeTypes.STATE:
                    all_read.append(v[0])
                values.append(ast.Name(id=v[0]))

            if len(values)>1:
                prev = None
                for v in values:
                    if prev:
                        prev = ast.BinOp(op=ast.Add(), left=v, right=prev)
                    else:
                        prev=v


                assign = ast.Assign(targets=[ast.Name(id=target_edges[0][1])], value=prev)
            else:
                assign = ast.Assign(targets=[ast.Name(id=target_edges[0][1])],
                                    value=values[0])


            body.append(assign)

    generate_program(equation_graph)

    all_must_init = set(all_read).difference(all_targeted)
    print('Must init: ',all_must_init)






    vars_node_id = {n[2]: n[0] for n in equation_graph.get_nodes() if n[2]}
    scope_var_node = {n[0]: n[1].scope_var for n in equation_graph.get_nodes() if n[2]}
    #print(scope_var_node)
    #asdasd=sdfsdfsdfsdf
    states = []
    deriv = []
    mapping = []
    other = []
    for sv_id, sv in scope_variables.items():
        if sv.type == VariableType.STATE:
            states.append(vars_node_id[sv.id])
        elif sv.type == VariableType.DERIVATIVE:
            deriv.append(vars_node_id[sv.id])
        #elif sv.sum_mapping_ids or sv.mapping_id:
        #    mapping.append(vars_node_id[sv.id])
        #else:
            #other.append(vars_node_id[sv.id])





    vars_init = states.copy()
    vars_init += list(all_must_init.difference(vars_init))

    vars_update = deriv.copy()
    vars_update += list(set(all_targeted).difference(vars_update))

    variables = vars_init + vars_update

    body = [
        ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id=i) for i in vars_init[len(states):]])], value=ast.Subscript(slice=ast.Slice(lower=ast.Num(n=len(states)), upper=ast.Num(n=len(vars_init)), step=None), value=ast.Name(id='variables'))),
        ast.Assign(targets=[ast.Tuple(elts=[ast.Name(id=s) for s in states])], value=ast.Name(id='y')),
        #ast.Assign(targets=[ast.Name(id=d) for d in deriv], value=ast.Num(n=0))
           ] + body

    body.append(ast.Assign(targets=[ast.Subscript(slice=ast.Slice(lower=ast.Num(n=len(vars_init)), upper=ast.Num(n=len(vars_init)+len(vars_update)), step=None),
                                               value=ast.Name(id='variables'))],
                           value=ast.Tuple(elts=[ast.Name(id=u) for u in vars_update])))

    body.append(ast.Return(value=ast.Call(func=ast.Name(id='np.array'), args=[ast.List(elts=[ast.Name(id=d) for d in deriv]), ast.Name(id='np.float64')], keywords=[])))

    kernel_args = dot_dict(args=[ast.Name(id='variables'), ast.Name(id='y')], vararg=None, defaults=[], kwarg=None)

    mod_body.append(wrap_function('kernel', body, decorators=["njit('float64[:](float64[:],float64[:])')"], args=kernel_args))

    source = generate_code_file(mod_body, 'kernel.py')
    print('compiling...')

    import timeit
    print('Compile time: ', timeit.timeit(
        lambda: exec(source, globals()), number=1))

    variables_values = np.array([scope_var_node[v].value for v in variables], dtype=np.float64)
    for k, v in zip(variables, variables_values):
        print(k,': ',v)

    return kernel, variables_values, variables