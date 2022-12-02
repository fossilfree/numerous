import logging

from .module import Module, ModuleSpec
from ..engine.system.subsystem import Subsystem
from ..multiphysics.equation_base import EquationBase
from ..multiphysics.equation_decorators import add_equation
from .variables import MappingTypes, Variable, Constant, StateVar, Derivative
from .connector import Directions


_logger = logging.getLogger("system generator")


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

def process_items_spec(host_system, host_module, items_spec_name, items_spec, tabs):
    items_spec = getattr(host_module, items_spec_name)

    #Check module specs have been instanciated
    modules = items_spec.get_modules()

    systems = {module_name: generate_system(module_name, module, tabs) for module_name, module in modules.items()}

    host_system.register_items(systems.values())

    items = Items(systems)

    setattr(host_system, items_spec_name, items)

def process_connector(name, connector):

    connected, map = connector.connection_map
    directions = connector.channel_directions
    if map:
        for other_key, self_key in map.items():
            direction = directions[other_key]

            other_obj = connected.channels[other_key]
            self_obj = connector.channels[self_key]

            to_obj = other_obj if direction == Directions.SET else self_obj
            from_obj = other_obj if direction == Directions.GET else self_obj

            if isinstance(from_obj, Variable) and isinstance(to_obj, Variable):
                to_obj.add(from_obj)
            elif isinstance(from_obj, (Module,ModuleSpec)) and isinstance(from_obj, (Module, ModuleSpec)):
                setattr(to_obj._parent.parent, to_obj._parent.attr, from_obj)
            else:
                raise ValueError(f"Expected either Module or Variable, not {from_obj} and {to_obj}")
    else:
        # TODO improve this error handling
        ValueError("no map!?")

def process_connectors(module:Module):
    # process connectors on this module
    connectors = module.get_connectors()

    for connector_name, connector in connectors.items():
        process_connector(connector_name, connector)

    #process recursive

        # Go through items specs
        for items_spec_name, items_spec in module.get_items_specs().items():
            modules = items_spec.get_modules(check=False)

            for module_name, module in modules.items():


                process_connectors(module)

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

    _logger.debug(tab_str(tabs, f'initializing generation of system <{name}>'))
    tabs_int = tabs + ["\t"]
    system = Subsystem(tag=name)

    process_connectors(module)


    # Go through items specs
    for items_spec_name, items_spec in module.get_items_specs().items():
        _logger.debug(tab_str(tabs_int, f'processing item spec <{items_spec_name}>'))
        process_items_spec(system, module, items_spec_name, items_spec, tabs=tabs_int+["\t"])

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

        for var_name, variable in variables.items():

            for _id, mapping, mapping_type in variable.mapping_with_types:
                if mapping.native_ref is None:
                    raise TypeError("!")
                if mapping_type == MappingTypes.ADD:
                    variable.native_ref.add_sum_mapping(mapping.native_ref)
                elif mapping_type == MappingTypes.ASSIGN:
                    variable.native_ref.add_mapping(mapping.native_ref)

                _logger.debug(tab_str(tabs_int, f'mapping <{mapping}>'))

        _logger.debug(tab_str(tabs_int, f"created mappings"))







    _logger.debug(tab_str(tabs, f'completed generation of system <{name}>'))

    return system
