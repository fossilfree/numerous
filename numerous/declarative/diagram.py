from graphviz import Digraph

from .specification import Module


class VariableNode:

    def __init__(self, var):
        self.variable = var
        self.edges = []
        self.id = var.id

    def add_edge(self, edge):
        self.edges.append(edge)


class ModuleGraph:

    def __init__(self, module, name):
        self.module = module
        self.modules = []
        self.name = name
        self.variables = {}

    def add_module(self, module):
        self.modules.append(module)

    def add_variable(self, variable: VariableNode):
        self.variables[variable.id] = variable

    def remove_variable(self, variable: VariableNode):
        self.variables.pop(variable.id)


def generate_module_subgraph(module, name, variables, modules, max_sub_level=1, sub_level=1, include_variables=False):
    mg = ModuleGraph(module, module.tag)

    scopes = {}
    if isinstance(module, Module):
        scopes = module._scope_specs
    else:
        if hasattr(module, '_namespaces'):
            scopes = module._namespaces

    for scope_name, scope_spec in scopes.items():
        # with c.subgraph(name='cluster_' +  str(uuid4())) as c_s:
        #    c_s.attr(label=scope_name)
        scope = getattr(module, scope_name)
        for var_name in scope_spec._variables.keys():
            var = getattr(scope, var_name)
            var_node = VariableNode(var)
            mg.add_variable(var_node)
            variables[var_node.id] = var_node

    if sub_level <= max_sub_level:
        # for item_spec_name, item_spec in module._item_specs.items():
        # with c.subgraph(name='cluster_' + str(uuid4())) as c_is:
        # c_is.attr(label=item_spec_name)
        for item_name, item in module.registered_items.items():
            #    for item_name, item in item_spec._items.items():
            # with c_is.subgraph(name='cluster_' + item_name) as c_i:
            #    c_i.attr(label=item_name)
            # item = getattr(item_spec, item_name)
            # if item not in modules:
            # modules.append(item)
            sub_mg = generate_module_subgraph(item, item_name, variables, modules=modules, max_sub_level=max_sub_level,
                                              sub_level=sub_level + 1)
            mg.add_module(sub_mg)
    return mg


"""
def generate_module_subgraph(g, module, name, variables, max_sub_level=1, sub_level=1, include_variables=False):
    with g.subgraph(name='cluster_' + str(uuid4())) as c:

        #c.attr(style='filled', color='lightgrey')
        #c.node_attr.update(style='filled', color='white')
        # c.edges([('a0', 'a1'), ('a1', 'a2'), ('a2', 'a3')])
        c.attr(label=name)
        #if include_variables:
        scopes = {}
        if isinstance(module, Module):
            scopes = module._scope_specs
        else:
            if hasattr(module,'_namespaces'):
                scopes = module._namespaces


        for scope_name, scope in scopes.items():
            #with c.subgraph(name='cluster_' +  str(uuid4())) as c_s:
            #    c_s.attr(label=scope_name)
                c_s = c
                for var_name, var in scope._variables.items():

                    c_s.node(var.id, label=var_name)
                    variables[var.id] = var


        if sub_level <= max_sub_level:
            for item_spec_name, item_spec in module._item_specs.items():
                #with c.subgraph(name='cluster_' + str(uuid4())) as c_is:
                    #c_is.attr(label=item_spec_name)
                    c_is = c
                    for item_name, item in item_spec._items.items():

                        #with c_is.subgraph(name='cluster_' + item_name) as c_i:
                        #    c_i.attr(label=item_name)
                        generate_module_subgraph(c_is, item, item_name, variables, max_sub_level=max_sub_level, sub_level=sub_level+1)
"""


def generate_graphviz_module(g, mg: ModuleGraph, variables_with_edges):
    with g.subgraph(name='cluster_' + mg.module.id) as sg:

        sg.attr(label=mg.name)

        for variable in mg.variables.values():
            if variable.id in variables_with_edges:
                sg.node(variable.id, label=variable.variable.tag)

        for module in mg.modules:
            generate_graphviz_module(sg, module, variables_with_edges)


def generate_graphviz(g, mg: ModuleGraph, variables_with_edges, edges):
    generate_graphviz_module(g, mg, variables_with_edges)

    cluster_combinations = []
    for to_var_node, from_var_node in edges:

        cluster_to = 'cluster_' + to_var_node.item.id
        cluster_from = 'cluster_' + from_var_node.item.id
        cluster_combo = cluster_from + cluster_to
        cluster_combo_reverse = cluster_to + cluster_from

        if not cluster_combo in cluster_combinations:
            cluster_combinations.append(cluster_combo)
            cluster_combinations.append(cluster_combo_reverse)

            g.edge(from_var_node.id, to_var_node.id, ltail=cluster_from, lhead=cluster_to)


def generate_diagram(module: Module, view=False, sub_levels=1, include_variables=False):
    variables = {}
    g = Digraph('G', filename='cluster.gv')
    # g.attr(splines="ortho")
    # g.attr(splines="ortho")

    # g.engine="osage"

    g.attr(compound='true')
    g.attr(sep='5')

    modules = []
    mg = generate_module_subgraph(module, module.tag, variables, modules=modules, max_sub_level=sub_levels, sub_level=1,
                                  include_variables=include_variables)

    variables_with_edges = []
    edges = []

    for id, var_node in variables.items():
        mapping = [var_node.variable.mapping] if var_node.variable.mapping else []

        for var_to in var_node.variable.sum_mapping + mapping:
            if var_to.id in variables:
                var_to_node = variables[var_to.id]
                edge = (var_node.variable, var_to_node.variable)
                edges.append(edge)
                var_node.add_edge(edge)
                var_to_node.add_edge(edge)
                variables_with_edges.append(id)
                variables_with_edges.append(var_to.id)
                # g.edge(var_to.id, var.id)
                # print(var_to, '>>',var)

    variables_with_edges = set(variables_with_edges)
    generate_graphviz(g, mg, variables_with_edges, edges)

    if view:
        g.view()

    # print()
    # print('v')
    # print(variables)
    # print('ve')
    # print(variables_with_edges)
    return g
