from .module import Module


def print_var(engine_var, get_val):
    print(f"{engine_var.path.primary_path}: {get_val(engine_var)}")


def print_all_module(module, get_val):
    if isinstance(module, Module):
        scopes = module.get_scope_specs()
    # elif isinstance(module, ModuleSpec):
    #    scopes = module._namespaces
    else:
        scopes = {}

    for scope_name, scope_spec in scopes.items():

        for name, variable in scope_spec.get_variables().items():
            engine_var = variable.native_ref
            print_var(engine_var, get_val)

    for items_spec_name, items_spec in module.get_items_specs().items():

        for module_name, sub_module in items_spec.get_modules(check=False).items():
            # sub_module = getattr(module, module_name)
            if isinstance(sub_module, Module):
                print_all_module(sub_module, get_val)


def print_all_variables(module, df):
    def get_val(var):
        return df[var.path.primary_path].tail(1).values[0]

    print_all_module(module, get_val)
