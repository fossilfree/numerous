from __future__ import annotations

from uuid import uuid4
import dataclasses
import inspect
from .context_managers import _active_mappings, _active_subsystem, NoManagerContextActiveException
from .exceptions import MappingOutsideMappingContextError, ItemNotAssignedError, NoSuchItemInSpec, MappingFailed, \
    NotMappedError, FixedMappedError
from .mappings import Obj, MappingTypes, ModuleMappings
from .variables import Variable, Operations, PartialResult
from numerous.declarative.bus import Connector, ModuleConnections
from .module import ModuleSpecInterface
from .utils import recursive_get_attr, RegisterHelper

from numerous.engine.system import Subsystem
from numerous.engine.variables import VariableType
from numerous.multiphysics.equation_base import EquationBase
from numerous.multiphysics.equation_decorators import add_equation


class ScopeSpec:
    """
       Specification of a scope in a module. Extend this class and create class variables to define a new specifcation of the variables in a namespace for a module.
       Variables should be added as class variables which will be discovered when instanciating the ScopeSpec and assigning it to a Module class.
   """

    def __init__(self, parent_path=[], new_var_ids=False):

        super(ScopeSpec, self).__init__()

        self._variables = {}
        self._host: Module = None
        self._host_attr = None

        self._equations = []

        class_var = {}

        for b in self.__class__.__bases__:
            class_var.update(b.__dict__)

        class_var.update(self.__class__.__dict__)

        for k, v in class_var.items():

            if isinstance(v, Variable):
                # instance = v.instance(id=".".join(parent_path+[k]), name=k)
                instance = v.instance(id=str(uuid4()) if new_var_ids else v.id, name=k, host=self)
                instance.set_host(self, k)
                #instance = v.instance(id=str(uuid4()), name=k, host=self)
                setattr(self, k, instance)

                self._variables[k] = instance

    def get_child_attr(self, child):
        for attr, val in self.__dict__.items():
            if val == child:
                return attr

        raise ModuleNotFoundError(f"{child} not an attribute on {self}")
    def get_path(self, parent):

        if self == parent:
            path = [self._host_attr]
            return path
        else:
            if inspect.isclass(self._host):
                path = self._host.get_path_class(parent) + [self._host_attr]
            else:
                path = self._host.get_path(parent) + [self._host_attr]
            return path
    def __setattr__(self, key, value, clone=False):

        if clone:
            super(ScopeSpec, self).__setattr__(key, value)
            return
        elif isinstance(value, PartialResult) and not isinstance(value, Variable):
            if value.op == Operations.ADD and value.arguments[0] == getattr(self, key) and isinstance(
                    value.arguments[1],
                    Variable):
                try:
                    mapping_manager = _active_mappings.get_active_manager_context()
                    mapping_manager.add(value.arguments[0], value.arguments[1])
                except NoManagerContextActiveException:
                    raise MappingOutsideMappingContextError(
                        "Mapping outside mapping context not possible. Place mappings inside a mapping context using 'create_mappings' context manager.")

            else:
                raise ValueError('Partial result not matching additive mapping.')
        elif hasattr(self, '_variables') and key in self._variables:
            try:
                mapping_context: ModuleMappings = _active_mappings.get_active_manager_context()
            except NoManagerContextActiveException:
                raise MappingOutsideMappingContextError("Mapping outside mapping context not possible. Place mappings inside a mapping context using 'create_mappings' context manager.")
            map_to = getattr(self, key)
            mapping_context.assign(map_to, value)

        else:
            super(ScopeSpec, self).__setattr__(key, value)

    def _instance(self, parent_path=[]):
        return self.__class__(parent_path)

    def _clone(self, host=None):
        clone = self.__class__(new_var_ids=True)
        clone._equations = self._equations
        clone.set_host(host, self._host_attr)
        # var_list = list(clone.__dict__.items())
        # for var, val in var_list:
        #    if isinstance(val, Variable):
        #        clone.__setattr__(var, val.clone(id=str(uuid4()), name=val.name), clone=True)

        return clone

    def set_host(self, host, attr):
        self._host = host
        self._host_attr = attr
    def _generate_variables(self, equation):
        ...

    def finalize(self):

        for eq in self._equations:
            eq.finalize()


class ItemsSpec:
    """
    Specification of the items in a module. Extend this class and create class variables to define a new specifcation of items for a module.
    Items or Modules are added as type hints or by assigning to class variables.
    """

    _initialized = False

    def __init__(self):

        super(ItemsSpec, self).__init__()

        self._items = {}
        self._host = None
        self._host_attr = None

        annotations = self.__class__.__annotations__

        for var, hint in annotations.items():
            if not hasattr(self, var):

                self._handle_item(var, hint)


        vallist = list(self.__class__.__dict__.items())# + list(self._items.items())
        for var, val in vallist:
            self._handle_item(var, val)

        self._initialized = True

    def set_host(self, host, attr):
        self._host = host
        self._host_attr = attr
        for name, item in self._items.items():
            if isinstance(item, Module) or isinstance(item, ModuleSpec):
                item.set_host(self, name)
    def get_child_attr(self, child):
        for attr, val in self.__dict__.items():
            if val == child:
                return attr

        raise ModuleNotFoundError(f"{child} not an attribute on {self}")

    def get_path(self, parent):

        _attr = self._host_attr

        if self == parent:

            path = [_attr]
            return path
        else:

            path = self._host.get_path(parent) + [_attr]
            return path

    def _clone(self):

        clone_ = self.__class__()
        #clone_._items = {k: v.clone() for k, v in self._items.items()}
        clone_._items = self._items.copy()
        for name, item in clone_._items.items():
            setattr(clone_, name, item)
        clone_.set_host(self._host, self._host_attr)
        return clone_

    def _handle_item(self, var, val):

        if isinstance(val, Module):
            self._items[var] = val

            raise NotImplementedError()

        elif inspect.isclass(val) and issubclass(val, Module) or val.__class__.__name__== "_AnnotatedAlias":

            clone_ = val.clone_tree()

            setattr(self, var, clone_)
            self._items[var] = clone_
            clone_.set_host(self, var)

        elif isinstance(val, ModuleSpec):
            raise NotImplementedError()
        #else:
        #    raise UnrecognizedItemSpec(f"The class {val.__class__.__name__} cannot be used as an item for {var}.")

    def _check_assigned(self):
        for name, item in self._items.items():
            item_ = getattr(self, name)

            if not isinstance(item_, Module) and item_._assigned_to is not None:
                a = item_._assigned_to

            if inspect.isclass(item_):
                raise ItemNotAssignedError(f'Item <{name}> is not assigned properly, value is a class {item_}.')
            elif isinstance(item_, ModuleSpec) and (item_._assigned_to is None or isinstance(item_._assigned_to, ModuleSpec)):
                raise ItemNotAssignedError(f'Item <{name}> on <{self._host.tag if self._host is not None else ""}> is still a {item_.__class__} object, meaning it as not been assigned during Module initialization.')
            elif item_ is None:
                raise ItemNotAssignedError(f'Item <{name}> is not assigned, value is {item_}.')
            elif not isinstance(item_, Module) and (not hasattr(item_, '_assigned_to') or item_._assigned_to is None):
                raise ItemNotAssignedError(f"The item <{item_}> is not a Module but instead <{item_.__class__.__name__}>")



    def finalize(self):
        """
        Finalize this module. Checks if all items from item specifications have been assigned.
        """
        for name, item in self._items.items():


            item_ = getattr(self, name)

            if isinstance(item_, Module):
                item_.finalize(top=False)
            if isinstance(item_, ModuleSpec):
                item_.finalize()
                if item_._assigned_to is not None:
                    setattr(self, name, item_._assigned_to)

        self._check_assigned()


    def __setattr__(self, key, value):
        if value is None and key not in ["_items", '_host', '_host_attr']:
            raise ItemNotAssignedError("You can not assign an item to None")

        if self._initialized and key not in ["_items", '_host', '_host_attr'] and not hasattr(self,key):
            raise NoSuchItemInSpec(f"ItemsSpec {self.__class__.__name__} class has no item {key}. Please add the item to the ItemsSpec class.")

        elif self._initialized and key!="_items" and isinstance(ms := getattr(self, key), ModuleSpec)  and key not in ["_items", '_host', '_host_attr']:
            ms._assigned_to = value

            if not isinstance(value, Module) and not isinstance(value, ModuleSpec):

                raise TypeError(f"Can only assign modules as item. Not: {value}")
            value._from_spec = ms
            value._host = self
            value._host_attr = key
        #Transfer connectors

        super(ItemsSpec, self).__setattr__(key,value)


class ModuleSpec(ModuleSpecInterface):
    """
    This class is used to represent a module as a specification added to an ItemsSpec, before it is replaced by an instanciated Module in the instanciation of a parent module.
    """
    def __init__(self, module_cls: object):

        super(ModuleSpec, self).__init__()



        self.module_cls = module_cls
        self._item_specs = {}
        self._namespaces = {}
        self._connectors = {}
        self._finalized = False
        self._host = None
        self._host_attr = None
        self.tag = module_cls.tag

        self._assigned_to = None

        class_var = {}

        for b in self.module_cls.__class__.__bases__:
            class_var.update(b.__dict__)

        class_var.update(self.module_cls.__dict__)

        for k, v in class_var.items():

            if isinstance(v, ItemsSpec):
                v.set_host(self, k)
                clone_ = v._clone()
                clone_.set_host(self, k)
                self._item_specs[k] = clone_
                setattr(self, k, clone_)

            elif isinstance(v, ScopeSpec):
                clone_ = v._clone(host=self)
                clone_.set_host(self, k)
                self._namespaces[k] = clone_
                setattr(self, k, clone_)

            elif isinstance(v, Connector):
                clone_ = v.clone()
                clone_.set_host(self, k)
                self._connectors[k] = clone_
                setattr(self, k, clone_)


    def set_host(self, host, attr):
        self._host_attr = attr
        self._host = host
    def get_child_attr(self, child):
        for attr, val in self.__dict__.items():
            if val == child:
                return attr

        raise ModuleNotFoundError(f"{child} not an attribute on {self}")
    def get_path(self, parent):
        if self == parent:
            path = []
            return path
        else:
            path = self._host.get_path(parent) + [self._host_attr]

            return path

    def finalize(self):
        self._finalized = True

        for item_spec in self._item_specs.values():
            item_spec.finalize()

        for scope_spec in self._namespaces.values():
            ...


class EquationSpec:
    """
       Specification of an equation in a module. Use this as a decorator for your methods implementing the equations in a module.
   """
    def __init__(self, scope: ScopeSpec):
        """
            scope: instance of the scope specification to which this equation will be added.
        """

        super(EquationSpec, self).__init__()
        """
        Create an equation specification.

        Will add the equation to the scope passed as the argument
        """
        self.scope = scope
        self.func = None

    def __call__(self, func):
        self.func = func
        self.scope._equations.append(self)



        return self.func

@dataclasses.dataclass
class ResolveInfo:
    resolved: bool = False
    path: list = dataclasses.field(default_factory=lambda: list())
    name: str = ""

def resolve_variable_(obj, resolve_var: Variable) -> ResolveInfo:

    if isinstance(obj, Module) or isinstance(obj, Obj):
        if obj._scopes is not None:

            for scope_name, scope in obj._scopes.items():

                for var_name, var in scope._variables.items():

                    if var.id == resolve_var.id:

                        return ResolveInfo(resolved=True, path=[scope_name], name=var_name)

        if obj._item_specs is not None:

            for item_name, item in obj._item_specs.items():
                resolve = resolve_variable_(item, resolve_var)
                if resolve.resolved:
                    resolve.path = [item_name] + resolve.path
                    return resolve


    elif (inspect.isclass(obj) and issubclass(obj, Module)):
        for attr_name, attr_val in obj.__dict__.items():
            if isinstance(attr_val, ScopeSpec):

                for var_name, var in attr_val._variables.items():

                    if var.id == resolve_var.id:
                        return ResolveInfo(resolved=True, path= [attr_name], name=var_name)
            if isinstance(attr_val, ItemsSpec):
                resolve = resolve_variable_(attr_val, resolve_var)
                if resolve.resolved:
                    resolve.path = [attr_name] + resolve.path
                    return resolve


    elif isinstance(obj, ItemsSpec):
        for item_name, item in obj._items.items():
            resolve = resolve_variable_(item, resolve_var)
            if resolve.resolved:
                resolve.path = [item_name] + resolve.path
                return resolve

    elif isinstance(obj, ModuleSpec):

        for attr_name, attr_val in obj._namespaces.items():
            if isinstance(attr_val, ScopeSpec):

                for var_name, var in attr_val._variables.items():
                    if var.id == resolve_var.id:

                        return ResolveInfo(resolved=True, path=[attr_name], name=var_name)

        for item_name, item in obj._item_specs.items():

            resolve = resolve_variable_(item, resolve_var)
            if resolve.resolved:
                resolve.path = [item_name] + resolve.path
                return resolve


    else:
        raise MappingFailed('Not a compatible object ' + str(obj))

    #raise MappingFailed(
    #    f"Could not find variable with id <{resolve_var.id}> and name <{resolve_var.name}> in scope spec <{attr_name}> of module spec <{obj.module_cls.__name__}>")

    return ResolveInfo()

def resolve_variable(obj, resolve_var: Variable) -> ResolveInfo:
    """
    Method to resolve the path to a variable on an object. It will work recursively to find the obj
    """
    resolve = resolve_variable_(obj, resolve_var)

    if not resolve.resolved:
        raise MappingFailed(f"Could not find <{resolve_var.id}, {resolve_var.name}> in <{obj}>")
    return resolve


class Module(Subsystem, EquationBase):

    """
        Module for builduing combined subsystems and equations using a declarative approach.

        Variables are declared as ScopeSpecs defined as classes
        Items/submodules are declared in ItemsSpecs classes using class variables
        Connectors are added as class variables
        mappings and connections are declared using create_mappings and create_connections context managers.
    """

    tag = None
    _finalized = False
    _mapping_groups = {}
    _item_specs = {}
    _scope_specs = {}
    _connectors = {}
    _resolved_mappings = []
    _must_map_variables = []
    _fixed_variables = []
    _scopes = {}
    _from_spec = None

    def __repr__(self):
        return object.__repr__(self)

    def __new__(cls, *args, **kwargs):
        parent_module = _active_subsystem.get_active_manager_context(ignore_no_context=True)

        register_helper = RegisterHelper()

        items_specs = {}
        scope_specs = {}
        mapping_groups = {}
        connectors = {}
        connections = {}
        _objects = {}

        base_scopes = {}
        # Get inherited equations
        for b in list(cls.__bases__) + [cls]:
            for var, val in b.__dict__.items():
                if isinstance(val, ScopeSpec):
                    if var in base_scopes:
                        for eq in base_scopes[var]._equations:
                            val._equations.append(eq)
                        base_scopes[var] = val
                    else:
                        base_scopes[var] = val

        for b in list(cls.__bases__) + [cls]:
            _objects.update(b.__dict__)



        org_init = cls.__init__

        def wrap(self, *args, **kwargs):
            for attr, var in base_scopes.items():
                if isinstance(var, ScopeSpec):
                    var.set_host(self, attr)
                    scope_specs[attr] = var
                    # watcher.add_watched_object(var)
                    # [watcher.add_watched_object(e) for e in var._equations]
            for attr, var in _objects.items():
                if isinstance(var, ItemsSpec):
                    var.set_host(self, attr=attr)
                    clone_ = var._clone()
                    items_specs[attr] = clone_

                    # watcher.add_watched_object(var)
                    # watcher.add_watched_object(clone_)
                    ...



                elif isinstance(var, Connector):
                    var.set_host(self, attr)
                    clone_ = var.clone()
                    connectors[attr] = clone_

                elif isinstance(var, ModuleConnections):
                    var.set_host(self, attr)
                    clone_ = var.clone()
                    connections[attr] = clone_

            _resolved_mappings = []
            for attr, var in _objects.items():

                if isinstance(var, ModuleMappings):
                    mapping_groups[attr] = var
                    # watcher.add_watched_object(var)

                    for mapping_ in var.mappings:
                        mapto = mapping_[0]
                        mapfrom = mapping_[1]

                        map_type = mapping_[2]

                        resolved_to = ResolveInfo(True, mapto.get_path(cls), "")
                        to_ = ".".join(resolved_to.path)

                        resolved_from = ResolveInfo(True, mapfrom.get_path(cls), "")

                        from_ = ".".join(resolved_from.path)

                        _resolved_mappings.append((resolved_to, resolved_from, map_type))

            self._resolved_mappings = _resolved_mappings
            self._must_map_variables = []
            self._fixed_variables = []
            self._scopes = {}
            self._scope_specs = scope_specs
            self._item_specs = items_specs
            self._connectors = connectors
            self._connections = connections

            for item_name, item in items_specs.items():
                setattr(self, item_name, item)

            self._mapping_groups = mapping_groups
            org_init(self, *args, **kwargs)

            cls.__init__ = org_init
            _active_subsystem.clear_active_manager_context(register_helper)
            _active_subsystem.set_active_manager_context(parent_module)

            if cls.tag is not None and instance.tag is None:
                instance.tag = cls.tag

            if instance.tag is None:
                instance.tag = cls.__name__.lower()

            [instance.register_item(i) for i in register_helper.get_items().values()]

            if isinstance(parent_module, RegisterHelper) or isinstance(parent_module, Subsystem):
                parent_module.register_item(instance)

            instance._item_specs = items_specs
            instance._scope_specs = scope_specs
            instance._mapping_groups = mapping_groups

        _active_subsystem.clear_active_manager_context(parent_module)
        _active_subsystem.set_active_manager_context(register_helper)
        cls.__init__ = wrap
        instance = object.__new__(cls)


        # watcher.add_watched_object(instance)

        return instance

    def __init__(self, *args, **kwargs):
        super(Module, self).__init__(*args, **kwargs)

        self._host = None
        self._host_attr = None

        #self._scope_specs = self._scope_specs.copy()
        for k, v in self._scope_specs.items():
            clone_ = v._clone(host=self)
            clone_.set_host(self, k)
            #v.attach()
            #watcher.add_watched_object(clone_)
            setattr(self, k, clone_)

            #    v.check_assigned()
            self.add_scope(self, k, clone_, kwargs)


    def check_variables(self):

        for v in self._must_map_variables:

            resolve = resolve_variable(self, v)

            var = getattr(recursive_get_attr(self, resolve.path), resolve.name)

            if var.mapping is None:
                raise NotMappedError(f"The variable {var.path.primary_path} is set must be mapped, but is not mapped!")

        for v in self._fixed_variables:
            resolve = resolve_variable(self, v)
            var = getattr(recursive_get_attr(self, resolve.path), resolve.name)

            if var.mapping is not None:
                raise FixedMappedError(
                    f"The variable {var.path.primary_path} is fixed, but mapped to another variable!")

    def _finalize(self, top=True):
        if not self._finalized:

            for connections_name, connections in self._connections.items():
                connections.finalize()

            # self._scope_specs.update(self.__dict__)
            for item in self.registered_items.values():
                item.finalize(top=False)

            for itemspec_name, itemspec in self._item_specs.items():
                itemspec.finalize()

            for resolved_mapping in self._resolved_mappings:
                resolved_from = resolved_mapping[1]
                resolved_to = resolved_mapping[0]
                map_type = resolved_mapping[2]
                if map_type == MappingTypes.ASSIGN:
                    try:

                        to_attr = recursive_get_attr(self, resolved_to.path)

                    except AttributeError as ae:
                        raise MappingFailed(f"!")

                    try:
                        from_attr = getattr(recursive_get_attr(self, resolved_from.path[:-1]),
                                            resolved_from.path[-1])
                    except AttributeError as ae:
                        raise MappingFailed(
                            f"The obj_ect {self}, {resolved_from.path} has no variable {resolved_from.name}")
                    except IndexError:
                        raise MappingFailed(
                            f"The obj_ect {self}, {resolved_from.path} has no variable {resolved_from.name}")

                    to_attr.add_mapping(from_attr)

                elif map_type == MappingTypes.ADD:
                    try:
                        to_attr = recursive_get_attr(self, resolved_to.path)
                    except AttributeError as ae:

                        raise MappingFailed(
                            f"The namespace {'.'.join([str(s) for s in [self] + resolved_to.path[:-1]])} does not have a variable {resolved_to.path[-1]}.")

                    try:
                        from_attr = getattr(recursive_get_attr(self, resolved_from.path[:-1]),
                                            resolved_from.path[-1])
                    except AttributeError as ae:
                        raise MappingFailed(
                            f"The obj_ect {self}, {resolved_from.path} has no variable {resolved_from.name}")

                    to_attr.__iadd__(from_attr)

                else:
                    raise ValueError('Unknown mapping type: ', map_type)



            self.check_variables()

            self._finalized = True

    def prefinalize(self):
        ...

    def finalize(self, top=True):

        self.prefinalize()
        self._finalize(top=top)


    @classmethod
    def add_scope(cls, obj, namespace, scope, values: dict):

        ns = obj.create_namespace(namespace)
        obj._scopes[namespace] = scope
        setattr(obj, namespace, ns)

        class Eq(EquationBase):

            tag = namespace + '_eq'

        eq_ = Eq()

        for eq_spec in scope._equations:
            eq_func = eq_spec.func
            eq_func.__self__ = obj
            eq_func_dec = add_equation(eq_, eq_func)
            #eq_spec.attach()


        prescope = list(scope.__dict__.items())

        for v_name, v_ in prescope:

            if isinstance(v_, Variable):

                if v_name in values:
                    val = values.pop(v_name)

                else:
                    val = v_.value

                if v_.fixed:
                    obj._fixed_variables.append(v_)
                if v_.must_map:
                    obj._must_map_variables.append(v_)

                if v_.construct:
                    if v_.type == VariableType.PARAMETER:
                        var = eq_.add_parameter(v_name, val, logger_level=None, alias=None, integrate=v_.integrate)

                    else:
                        var = eq_.add_variable(v_name, val, var_type=v_.type, logger_level=None, alias=None)

                #    v_.set_variable(var)

        scope._generate_variables(eq_)
        ns.add_equations([eq_])

    @classmethod
    def clone_tree(cls):
        clone = ModuleSpec(cls)
        return clone

    def get_path(self, parent):

        if self == parent or self.__class__ == parent:
            return []
        else:
            path = self._host.get_path(parent) +[self._host_attr]
            return path