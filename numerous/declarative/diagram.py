from graphviz import Digraph
from .specification import Module
from uuid import uuid4

def generate_module_subgraph(g, module, name, variables):
    with g.subgraph(name='cluster_' + str(uuid4())) as c:

        #c.attr(style='filled', color='lightgrey')
        #c.node_attr.update(style='filled', color='white')
        # c.edges([('a0', 'a1'), ('a1', 'a2'), ('a2', 'a3')])
        c.attr(label=name)

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



        for item_spec_name, item_spec in module._item_specs.items():
            #with c.subgraph(name='cluster_' + str(uuid4())) as c_is:
                #c_is.attr(label=item_spec_name)
                c_is = c
                for item_name, item in item_spec._items.items():
                    print(item_name)
                    #with c_is.subgraph(name='cluster_' + item_name) as c_i:
                    #    c_i.attr(label=item_name)
                    generate_module_subgraph(c_is, item, item_name, variables)

def generate_diagram(module: Module, view=False):
    variables = {}
    g = Digraph('G', filename='cluster.gv')
    g.attr(splines="ortho")
    generate_module_subgraph(g, module, module.tag, variables)

    for id, var in variables.items():
        for var_to in var.mapped_to:
            g.edge(var.id, var_to.id)
            print(var, '>>',var_to)

    if view:
        g.view()

    return g
