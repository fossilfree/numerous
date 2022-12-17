import logging
import uuid

from .module import Module, ModuleSpec, AutoItemsSpec
from ..engine.system.subsystem import Subsystem
from ..multiphysics.equation_base import EquationBase
from ..multiphysics.equation_decorators import add_equation
from .variables import MappingTypes, Variable, Constant, StateVar, Derivative
from .connector import Directions
from .mappings import create_mappings


_logger = logging.getLogger("system generator")

def path_str(path:list):
    return ".".join(path
                    )
def tab_str(tabs, text):
    return "".join(tabs + [text])


def process_module_items(module: Module|ModuleSpec, path, tabs):
    _logger.info(f'process module: {module}')
    for scope_name, scope in module.scopes.items():
        scope_path = path+(scope_name,)
        _logger.info(f'process scope: {scope}')

        for var_name, variable in scope.variables.items():
            _logger.info(f"Path for {variable}")
            if variable.path is None:
                #sdfsfd=dsfsdfdfg
                variable.path = scope_path + (var_name,)

                _logger.info(f"Setting path of {path_str(variable.path)}")
            else:
                _logger.info(f"Already have path: {path_str(variable.path)}")
                _logger.info(f"Wanted to set {path_str(scope_path + (var_name,))}")



    if hasattr(module,"module_spec") and module.module_spec:
        _logger.info(f"Setting path for 'spec' {path_str(path)}")
        process_module_items(module.module_spec, path, tabs)

    for itemsspec_name, itemsspec in module.items_specs.items():
        items_path = path+(itemsspec_name,)
        _logger.info(f"Processing {path_str(items_path)}, {itemsspec}")
        _logger.info(f"{itemsspec._modules.items()}")
        for sub_module_name, sub_module in itemsspec._modules.items():
            module_path = items_path+(sub_module_name,)
            _logger.info(f"Setting path for {path_str(module_path)}, {sub_module}")

            process_module_items(sub_module, module_path, tabs)

def process_module_connections(module_path: tuple[str], module: Module, tabs):
    all_connections = []
    _logger.info(tabs+f"Processing connections of module {path_str(module_path)}")
    for connection_set_name, connection_set in module.connection_sets.items():
        # Check all connections
        for connection in connection_set.connections:
            _logger.info(tabs + f"Processing connection {connection}")

            for side1_name, side1, side2_name, side2, direction in connection.channels:
                to_ = side1 if direction == Directions.GET else side2
                from_ = side2 if direction == Directions.GET else side1
                _logger.info(tabs + f"{side1_name} <<>> {side2_name}")
                if isinstance(to_, (ModuleSpec,)) and isinstance(from_, (ModuleSpec,)) and from_.assigned_to:
                    to_.assigned_to = from_.assigned_to

                elif isinstance(from_, Variable) and isinstance(to_, Variable):
                    to_.add_sum_mapping(from_)
                else:
                    raise TypeError('!')

    for itemsspec_name, itemsspec in module.items_specs.items():
        items_path = module_path+(itemsspec_name,)
        for sub_module_name, sub_module in itemsspec.get_modules(include_specs=True, ignore_not_assigned=True).items():
            sub_module_path = items_path + (sub_module_name,)

            process_module_connections(sub_module_path, sub_module, tabs+"\t")

def process_mappings(module: Module|ModuleSpec, path, objects, tabs, spec=False):
    _logger.info(f"Process for {path_str(path)} - spec: {spec}")

    for scope_name, scope in module.scopes.items():

        for variable_name, variable in scope.variables.items():
            #_logger.info(tabs+f"Mappings for {path_str(path+(scope_name, variable_name))}")
            var_path = path_str(variable.path)
            #_logger.info(f"Process mappings for {var_path}")

            for mapping in variable.mappings:
                #_logger.info(f"Process mapping {mapping[1]}")


                if mapping[1].path is None:
                    _logger.info(tabs + f"{var_path} << ?")
                map_path = path_str(mapping[1].path)
                _logger.info(tabs+f"{var_path} << {map_path}")

                var = objects[variable.path].native_ref
                mapping_var = objects[mapping[1].path].native_ref

                var.add_sum_mapping(mapping_var)

    if isinstance(module, Module) and module.module_spec:

        process_mappings(module.module_spec, path, objects, tabs, spec=True)

    for itemsspec_name, itemsspec in module.items_specs.items():
        items_path = path+(itemsspec_name,)
        for sub_module_name, sub_module in itemsspec._modules.items():

            sub_module_path = items_path + (sub_module_name,)

            #_logger.info(f"Process mappings for {sub_module_path}")

            process_mappings(sub_module, sub_module_path, objects, tabs+"\t")



def reprocess(module: Module, path: tuple, objects):
    for scope_name, scope in module.scopes.items():
        scope_path = path+(scope_name,)

        for var_name, variable in scope.variables.items():
            var_path = scope_path + (var_name,)
            objects[var_path] = variable

    for itemsspec_name, itemsspec in module.items_specs.items():
        items_path = path + (itemsspec_name,)
        for sub_module_name, sub_module in itemsspec.modules.items():
            submodule_path = items_path + (sub_module_name,)

            reprocess(sub_module, submodule_path, objects)

def generate_subsystem(name:str, module:Module, processed_modules, path, objects):

    system = Subsystem(name)

    objects[path] = system

    for scope_name, scope in module.scopes.items():

        scope_path = tuple(path + (scope_name,))

        #_logger.debug(tab_str(tabs_int, f"creating namespace <{scope_name}>"))
        namespace = system.create_namespace(scope_name)
        objects[scope_path] = namespace
        setattr(system, scope_name, namespace)

        #_logger.debug(tab_str(tabs_int, f"creating equation <{scope_name}>"))

        equation = create_equation(system, scope_name, scope.equations)

        for var_name, variable in scope.variables.items():

            if variable.value is None:
                raise TypeError(f"Variable {var_name} has not been assigned a value!")

            if isinstance(variable, StateVar):
                var_desc = equation.add_state(var_name, variable.value)
            elif isinstance(variable, Derivative):
                ...
            else:
                var_desc = equation.add_parameter(var_name, variable.value)

        namespace.add_equations([equation])

        for var_name, variable in scope.variables.items():
            variable.native_ref = getattr(namespace, var_name)
            var_path = scope_path + (var_name,)
            objects[var_path] = variable

    processed_modules.append(module)

    for itemsspec_name, itemsspec in module.items_specs.items():
        items_path = path + (itemsspec_name,)

        for sub_module_name, sub_module in itemsspec.modules.items():
            submodule_path = items_path + (sub_module_name,)
            if not sub_module in processed_modules:
                _logger.info(f"sub module: {path_str(submodule_path)}")
                sub_system = generate_subsystem(sub_module_name, sub_module, processed_modules, submodule_path, objects)

                system.register_item(sub_system)
            else:
                reprocess(sub_module, submodule_path, objects)

    return system

def create_equation(system, name, equation_specifications):
    class Eq(EquationBase):
        tag = name + "_eq"

    equation = Eq()

    for eq_spec in equation_specifications:
        eq_func = eq_spec.func
        eq_func.__self__ = system
        eq_func_dec = add_equation(equation, eq_func)

    return equation

def generate_system(name:str, module: Module):

    objects = {}
    path = (name,)
    processed_modules = []

    tabs = ""

    # Process connections - this ensures all modules are updated through connectors, before check of items specs assignments are complete.
    _logger.info("Processing model connections")

    process_module_connections(path, module, tabs+"\t")

    # Process items specs - create system hierarchy
    _logger.info("Processing model items")

    process_module_items(module, path, tabs)

    _logger.info("Generating numerous engine system")

    system = generate_subsystem(name, module, processed_modules, path, objects)

    # Process mappings - finally loop to ensure all mappings are in place
    _logger.info("Processing model mappings")
    process_mappings(module, path, objects, tabs)

    _logger.info(f'completed generation of system from model.')

    return system
