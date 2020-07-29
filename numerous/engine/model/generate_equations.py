from numerous.engine.model.graph import Graph
from numerous.engine.model.utils import NodeTypes, recurse_Attribute, wrap_function, dot_dict, generate_code_file
from numerous.engine.model.parser_ast import attr_ast, function_from_graph_generic

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


def generate_equations(equations, equation_graph: Graph, scoped_equations):
    mod_body = []
    #Loop over equation functions and generate code
    eq_vardefs={}
    for eq_key, eq in equations.items():
        vardef = Vardef()
        func, vardef_ = function_from_graph_generic(eq[2],eq_key.replace('.','_'), var_def_=vardef)
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
            print(n[0])
            args_local = [ae[0] for ae in equation_graph.edges_end(n, 'arg') if equation_graph.nodes_map[ae[0]][1].scope_var]
            all_read += args_local
            targets_local = [te[1] for te in equation_graph.edges_start(n, 'target') if equation_graph.nodes_map[te[1]][1].scope_var]
            all_targeted += targets_local
            scope_vars = {'scope.'+equation_graph.nodes_map[al][1].scope_var.tag: al for al in args_local + targets_local}
            print(scope_vars)


            args = [ast.Name(id=scope_vars[a]) for a in vardef.args]

            if len(vardef.targets)>1:
                targets = [ast.Tuple(elts=[ast.Name(id=scope_vars[t]) for t in vardef.targets])]
            else:
                targets = [ast.Name(id=scope_vars[vardef.targets[0]])]
            body.append(ast.Assign(targets=targets, value=ast.Call(func=ast.Name(id=scoped_equations[n[0]].replace('.','_')), args=args, keywords=[])))

        if n[1].node_type == NodeTypes.ASSIGN:
            target_edges = equation_graph.edges_start(n, 'target')
            value_edge = equation_graph.edges_end(n, 'value')
            all_targeted.append(target_edges[0][1])
            all_read.append(value_edge[0][0])
            assign = ast.Assign(targets=[ast.Name(id=target_edges[0][1])], value=ast.Name(id=value_edge[0][0]))


            body.append(assign)

    all_must_init = set(all_read).difference(all_targeted)
    print('Must init: ',all_must_init)


    kernel_args = dot_dict(args=[], vararg=None, defaults=[], kwarg=None)
    mod_body.append(wrap_function('kernel', body, decorators=[], args=kernel_args))


    generate_code_file(mod_body, 'kernel.py')
