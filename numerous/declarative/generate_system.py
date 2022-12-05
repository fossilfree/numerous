import logging
import uuid

from .module import Module, ModuleSpec, AutoItemsSpec
from ..engine.system.subsystem import Subsystem
from ..multiphysics.equation_base import EquationBase
from ..multiphysics.equation_decorators import add_equation
from .variables import MappingTypes, Variable, Constant, StateVar, Derivative
from .mappings import create_mappings
from .connector import Directions, Connection


_logger = logging.getLogger("system generator")

def path_str(path:list):
    return ".".join(path)

class Items(object):

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def create_equation(system, name, equation_specifications):
    class Eq(EquationBase):
        tag = name + "_eq"

    equation = Eq()

    for eq_spec in equation_specifications:
        eq_func = eq_spec.func
        eq_func.__self__ = system
        eq_func_dec = add_equation(equation, eq_func)

    return equation

def process_items_spec(host, host_system, host_module, items_spec_name, items_spec, tabs):
    _logger.debug(tab_str(tabs, f'processing item spec <{path_str(items_spec.get_path(host))}>'))

    items_spec = getattr(host_module, items_spec_name)

    if isinstance(items_spec, AutoItemsSpec):

        items_spec.remove_non_orphants()

        _logger.debug(tab_str(tabs, f"Auto detected {len(items_spec.modules)}."))

    #Check module specs have been instanciated
    modules = items_spec.get_modules()

    systems = {module_name: process_items_scopes(host, module_name, module, tabs) for module_name, module in modules.items() if isinstance(module, Module) and not module._processed and items_spec == module._parent.parent}

    host_system.register_items(systems.values())

    items = Items(systems)

    setattr(host_system, items_spec_name, items)

def recursive_get_attr(obj, attr_list):
    attr_ = getattr(obj, attr_list[0])

    if len(attr_list) > 1:
        return recursive_get_attr(attr_, attr_list[1:])
    else:
        return attr_


def process_connection(host: Module, module: Module, connection: Connection):
    #print(connection.side1.var1.get_path(host))
    #print(connection.side2.var1.get_path(host))


    map = connection.map
    directions = connection.side1.channel_directions
    if map:

        with create_mappings() as mappings:

            for side2_key, side1_key in map.items():
                direction = directions[side2_key]

                side1_obj = recursive_get_attr(host, connection.side1.channels[side1_key].get_path(host))
                side2_obj = recursive_get_attr(host, connection.side2.channels[side2_key].get_path(host))

                to_obj = side1_obj if direction == Directions.GET else side2_obj
                from_obj = side1_obj if direction == Directions.SET else side2_obj

                if isinstance(from_obj, Variable) and isinstance(to_obj, Variable):
                    to_obj.add(from_obj)
                elif isinstance(from_obj, (Module,ModuleSpec)) and isinstance(from_obj, (Module, ModuleSpec)):
                    setattr(to_obj._parent.parent, to_obj._parent.attr, from_obj)
                else:
                    raise ValueError(f"Expected either Module or Variable, not {from_obj} and {to_obj}")

        host.add_reference(str(uuid.uuid4()), mappings)
    else:
        # TODO improve this error handling
        ValueError("no map!?")

def process_connection_sets(host:Module, module:Module):
    # process connectors on this module
    connection_sets = module.get_connection_sets()

    for connection_set_name, connection_set in connection_sets.items():
        for connection in connection_set.connections:
            process_connection(host, module, connection)

    #process recursive

    # Go through items specs
    for items_spec_name, items_spec in module.get_items_specs().items():
        modules = items_spec.get_modules(check=False)

        for module_name, module in modules.items():
            if isinstance(module, Module):
                process_connection_sets(host, module)

def process_mappings(module: Module, host: Module,tabs_int:list):


    """for scope_name, scope in module.get_scope_specs().items():
        _logger.debug(tab_str(tabs_int, f"create mappings for {scope_name}"))
        for var_name, variable in scope.get_variables().items():
            variable_live = recursive_get_attr(host, variable.get_path(host))
            for _id, mapping, mapping_type in variable.mapping_with_types:
                mapping = recursive_get_attr(host, mapping.get_path(host))



                if mapping.native_ref is not None:
                    _logger.debug(tab_str(tabs_int,
                                          f'{variable_live.native_ref.path.primary_path} << {mapping.native_ref.path.primary_path}'))

                    if mapping_type == MappingTypes.ADD:
                        variable_live.native_ref.add_sum_mapping(mapping.native_ref)
                    elif mapping_type == MappingTypes.ASSIGN:
                        variable_live.native_ref.add_mapping(mapping.native_ref)
                    else:
                        raise TypeError(f"Unknown mapping type {mapping_type}")

                    #_logger.debug(tab_str(tabs_int, f'mapping <{mapping}>'))
                else:

                    _logger.debug(tab_str(tabs_int, str(mapping.get_path(host))))
                    _logger.debug(tab_str(tabs_int, str(recursive_get_attr(host, mapping.get_path(host)))))

                    _logger.debug(tab_str(tabs_int, f'Var has no native and no parent <{mapping}>, <{mapping._parent}>'))
                    raise TypeError(
                    f"Variable <{var_name}> with a <{mapping._parent.attr}> does not have a native reference set yet!")"""
    for mappings_name, mappings in module.get_mappings().items():

        for a, b, mapping_type in mappings.mappings:

            path_a = a.get_path(host)
            path_b = b.get_path(host)
            _logger.debug(tab_str(tabs_int, f"Mapping: {path_str(path_a)}<<{path_str(path_b)}"))


            var_a = recursive_get_attr(host, path_a)
            var_b = recursive_get_attr(host, path_b)

            _logger.debug(tab_str(tabs_int, f"Native: {var_a.native_ref.path.primary_path}<<{var_b.native_ref.path.primary_path}"))

            if mapping_type == MappingTypes.ADD:
                var_a.native_ref.add_sum_mapping(var_b.native_ref)
            elif mapping_type == MappingTypes.ASSIGN:
                var_a.native_ref.add_mapping(var_b.native_ref)
            else:
                raise TypeError(f"Unknown mapping type {mapping_type}")


        _logger.debug(tab_str(tabs_int, f"created mappings"))

    for items_spec_name, items_spec in module.get_items_specs().items():
        modules = items_spec.get_modules(check=False)

        for module_name, module in modules.items():
            if isinstance(module, Module):
                _logger.debug(tab_str(tabs_int, f"Process mappings for {path_str(module.get_path(host))}"))

                process_mappings(module, host, tabs_int)



def process_items_scopes(host, name, module, tabs_int):


    module._processed = True
    if not host==module:
        _logger.debug(tab_str(tabs_int, f'initializing generation of system <{path_str(module.get_path(host))}>'))

    system = Subsystem(tag=name)
    # Go through items specs
    items_specs = module.get_items_specs()


    process_items_spec(host, system, module, "unbound", items_specs.pop("unbound"), tabs=tabs_int + ["\t"])
    for items_spec_name, items_spec in items_specs.items():
        process_items_spec(host, system, module, items_spec_name, items_spec, tabs=tabs_int + ["\t"])



    # Make namespaces from scopespecs
    scopes = module.get_scope_specs()

    # Add variables

    for scope_name, scope in scopes.items():
        _logger.debug(tab_str(tabs_int, f"creating namespace <{scope_name}>"))
        namespace = system.create_namespace(scope_name)

        setattr(system, scope_name, namespace)

        _logger.debug(tab_str(tabs_int, f"creating equation <{scope_name}>"))

        equation = create_equation(system, scope_name, scope.equations)

        variables = scope.get_variables()

        for var_name, variable in variables.items():
            if isinstance(variable, StateVar):
                var_desc = equation.add_state(var_name, variable.value)
            elif isinstance(variable, Derivative):
                ...
            else:
                var_desc = equation.add_parameter(var_name, variable.value)

        _logger.debug(tab_str(tabs_int, f"added variables {list(variables.keys())}"))

        namespace.add_equations([equation])

        for var_name, variable in variables.items():
            variable.native_ref = getattr(namespace, var_name)

    return system

def tab_str(tabs, text):
    return "".join(tabs + [text])

""" 
    TODO: 
    - mappings
    - connectors
    - different variable types 
"""

def generate_system(name:str, module: Module, tabs=None):

    if tabs is None:
        tabs = []


    tabs_int = tabs + ["\t"]

    process_connection_sets(module, module)

    system = process_items_scopes(module, name, module, tabs_int)

    process_mappings(module, module, tabs_int)

    _logger.debug(tab_str(tabs, f'completed generation of system <{name}>'))

    return system
