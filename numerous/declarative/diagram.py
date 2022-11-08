from graphviz import Digraph
from .specification import Module
from uuid import uuid4

def generate_module_subgraph(g, module, name):
    with g.subgraph(name='cluster_' + str(uuid4())) as c:
        #c.attr(style='filled', color='lightgrey')
        #c.node_attr.update(style='filled', color='white')
        # c.edges([('a0', 'a1'), ('a1', 'a2'), ('a2', 'a3')])
        c.attr(label=name)

        if isinstance(module, Module):
            scopes = module._scope_specs
        else:
            scopes = module._namespaces

        for scope_name, scope in scopes.items():
            with c.subgraph(name='cluster_' + scope_name) as c_s:
                for var_name, var in scope._variables.items():
                    c_s.node(var_name)
                    c.attr(label=name)

        for item_spec_name, item_spec in module._item_specs.items():
            with c.subgraph(name='cluster_' + item_spec_name) as c_is:
                for item_name, item in item_spec._items.items():
                    with c_is.subgraph(name='cluster_' + item_name) as c_i:
                        generate_module_subgraph(c_i, item, item_name)

def generate_diagram(module: Module, view=False):

    g = Digraph('G', filename='cluster.gv')

    generate_module_subgraph(g, module, module.tag)

    if view:
        g.view()

    return g
