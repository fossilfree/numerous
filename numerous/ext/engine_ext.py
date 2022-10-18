from __future__ import annotations
import types
import typing
from typing import Annotated

from uuid import uuid4
from enum import Enum

from numerous.multiphysics.equation_base import EquationBase
from numerous.multiphysics.equation_decorators import add_equation

from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine.variables import VariableType

class AnotherManagerContextActiveException(Exception):
    pass

class NoManagerContextActiveException(Exception):
    pass

class ActiveContextManager:
    _instance = None

    def __new__(class_, *args, **kwargs):
        if not isinstance(class_._instance, class_):
            class_._instance = object.__new__(class_, *args, **kwargs)
        return class_._instance

    def __init__(self):
        self._active_manager_context = None

    def get_active_manager_context(self, ignore_no_context=False):
        if not ignore_no_context and not self.is_active_manager_context_set():
            raise NoManagerContextActiveException('No active context manager!')

        return self._active_manager_context

    def is_active_manager_context_set(self):
        return self._active_manager_context is not None

    def set_active_manager_context(self, context_manager):
        if self.is_active_manager_context_set():
            raise AnotherManagerContextActiveException('Another context active!')

        self._active_manager_context = context_manager

    def is_active_manager_context(self, context_manager):
        return self.get_active_manager_context(ignore_no_context=True) == context_manager

    def clear_active_manager_context(self, context_manager):
        if self.is_active_manager_context(context_manager):
            self._active_manager_context = None


class SubsystemContextManager(ActiveContextManager):
    pass

_active_subsystem = SubsystemContextManager()

class MappingsContextManager(ActiveContextManager):
    pass

_active_mappings = MappingsContextManager()

class Operations(Enum):

    ADD = 1
    SUB = 2
    DIV = 3
    MUL = 4
    POW = 5
    FUNC = 6
    NEG = 7
    LT = 8
    GT = 9
    GET = 10
    LET = 11
    EQ = 12

class PartialResult:

    def __init__(self, *args, op=None, func=None):
        self.id = str(uuid4())

        self.arguments = args

        self.op = op
        self.func = func





    def __add__(self, other):
        return PartialResult(self, other, op=Operations.ADD)

    def __sub__(self, other):
        return PartialResult(self, other, op=Operations.SUB)

    def __mul__(self, other):
        return PartialResult(self, other, op=Operations.MUL)

    def __truediv__(self, other):
        return PartialResult(self, other, op=Operations.DIV)

    def __neg__(self):
        return PartialResult(self, op=Operations.NEG)

    def __lt__(self, other):
        return PartialResult(self, other, op=Operations.LT)

    def __gt__(self, other):
        return PartialResult(self, other, op=Operations.GT)

    def __eq__(self, other):
        return PartialResult(self, other, op=Operations.EQ)


    def clone(self, variables):

        args_ = []
        for arg in self.arguments:
            if isinstance(arg, Variable):

                a = getattr(variables, arg.name)
            else:
                a = arg.clone(variables)

            args_.append(a)


        return PartialResult(*args_, op=self.op, func=self.func)



class Variable(PartialResult):

    def __init__(self, value, id=None, name=None, is_deriv=False, solve_for=True, instance=False,
                 vartype="Parameter", var_instance=None, integrate=None, construct=True):
        if id is None:
            id = str(uuid4())
        self.name = name
        self.id = id
        self.value = value
        self.is_deriv = is_deriv
        self.set_solve_for(solve_for)
        self.is_instance = instance
        self.cls = self
        self.integrate = integrate
        self._vartype = vartype
        self._var_instance = var_instance
        self.construct = construct


    def clone(self, id, name=None, instance=False):
        return Variable(id=id, value=self.value, name=self.name if name is None else name, is_deriv=self.is_deriv, solve_for=self.solve_for, instance=instance, vartype=self._vartype,
                        var_instance=self._var_instance, integrate=self.integrate, construct=self.construct)

    def instance(self, id, name):
        if self.is_instance:
            raise ValueError('Cannot instance from an instance')
        instance = self.clone(id=id, name=name, instance=True)
        instance.cls = self
        return instance


    def set_solve_for(self, solve_for: bool):
        self.solve_for = solve_for

    def __repr__(self):
        return f"{self.name}, {self.id}"

    def __eq__(self, other:Variable):
        if hasattr(other, 'id'):
            return self.id == other.id
        else:
            return False

class Parameter(Variable):
    def __init__(self, value, id=None, name=None, integrate=None):
        super(Parameter, self).__init__(value, id, name, is_deriv=False, solve_for=False, vartype="Parameter", integrate=integrate)

#class State(Variable):
#    def __init__(self, value, id=None, name=None):
#        super(State, self).__init__(value, id, name, is_deriv=False, solve_for=False, vartype="State")

class Constant(Variable):

    def __init__(self, value, id=None, name=None):
        super(Constant, self).__init__(value, id, name, is_deriv=False, solve_for=False, vartype="Constant")

    def set_solve_for(self, solve_for: bool):
        if solve_for:
            raise TypeError(f'Cannot solve for Constant variables. {self.name}, {self.id}')
        else:
            self.solve_for = solve_for

def State(value):

    return Variable(value, solve_for=False, vartype="State"), Variable(0, solve_for=True, vartype="deriv")

def integrate(var, integration_spec):
    var.integrate = integration_spec
    return var, Variable(0, construct=False)


var_type_map = {
    'Constant': VariableType.CONSTANT,
    'State': VariableType.STATE,
    'Parameter': VariableType.PARAMETER

}

def construct_placeholder(obj, cls):

    namespaces = {}

    for k, v in cls.__dict__._items():

        if '__' not in k:

            if isinstance(v, Module):

                pass
            #namespaces[k] =



#class ItemPlaceHolder:

#    def __init__(self, item_cls: Item):

#        self.namespaces = {}

#        for

class ModuleSpec:

    def __init__(self, module_cls: object):
        self.module_cls = module_cls
        self._item_specs = {}
        self._namespaces = {}
        # Find namespaces
        class_var = {}

        for b in self.module_cls.__class__.__bases__:
            class_var.update(b.__dict__)

        class_var.update(self.module_cls.__dict__)

        for k, v in class_var.items():


        #for k, v in self.module_cls.__dict__.items():
            if isinstance(v, ItemsSpec):
                clone_ = v.clone()
                self._item_specs[k] = clone_
                setattr(self, k, clone_)

            elif isinstance(v, ScopeSpec):
                clone_ = v.clone()
                self._namespaces[k] = clone_
                setattr(self, k, clone_)

    def finalize(self):
        ...

from typing import Annotated, Optional, get_type_hints

class ItemNotAssignedError(Exception):
    ...

class NoSuchItemInSpec(Exception):
    ...

class UnrecognizedItemSpec(Exception):
    ...

class ItemsSpec:

    _initialized = False

    def __init__(self):


        self._items = {}

        annotations = self.__class__.__annotations__
        #a = get_type_hints(self.__class__)
        for var, hint in annotations.items():
            if not hasattr(self, var):

                self.handle_item(var, hint)

            #self._items[var] = getattr(self, var)



        vallist = list(self.__class__.__dict__.items())# + list(self._items.items())
        for var, val in vallist:
            self.handle_item(var, val)

        self._initialized = True


    def clone(self):

        return self.__class__()

    def handle_item(self, var, val):

        if isinstance(val, Module):
            self._items[var] = val

            raise NotImplementedError()

        elif inspect.isclass(val) and issubclass(val, Module) or val.__class__.__name__=="_AnnotatedAlias":

            clone_ = val.clone_tree()

            setattr(self, var, clone_)
            self._items[var] = clone_

        elif isinstance(val, ModuleSpec):
            raise NotImplementedError()
        #else:
        #    raise UnrecognizedItemSpec(f"The class {val.__class__.__name__} cannot be used as an item for {var}.")

    def check_assigned(self):
        for name, item in self._items.items():
            item_ = getattr(self, name)
            if inspect.isclass(item_):
                raise ItemNotAssignedError(f'Item <{name}> is not assigned properly, value is a class {item_}.')
            elif isinstance(item_, ModuleSpec):
                raise ItemNotAssignedError(f'Item <{name}> is still a {item_.__class__} object, meaning it as not been assigned during Module initialization.')
            elif item_ is None:
                raise ItemNotAssignedError(f'Item <{name}> is not assigned, value is {item_}.')
            elif not isinstance(item_, Item):
                raise ItemNotAssignedError(f'Item <{name}> is not assigned to a numerous Item, value is {item_}.')

    def finalize(self):
        for item in self._items.values():
            if isinstance(item, Module) or isinstance(item, ModuleSpec):
                item.finalize()

        self.check_assigned()

    def __setattr__(self, key, value):
        if self._initialized and key!="_items" and not hasattr(self,key):
            raise NoSuchItemInSpec(f"ItemsSpec {self.__class__.__name__} class has no item {key}. Please add the item to the ItemsSpec class.")
        super(ItemsSpec, self).__setattr__(key,value)

def fake_init(*args, **kwargs):
    ...

def resolve_variable(obj, resolve_var: Variable):

    if isinstance(obj, Module):
        if obj._scopes is not None:

            for scope_name, scope in obj._scopes.items():

                for var_name, var in scope._variables.items():

                    if var.id == resolve_var.id:

                        return True, [scope_name], var_name

        if obj._item_specs is not None:

            for item_name, item in obj._item_specs.items():
                found, path_, var_name = resolve_variable(item, resolve_var)
                if found:
                    return found, [item_name] + path_, var_name

    elif (inspect.isclass(obj) and issubclass(obj, Module)):
        for attr_name, attr_val in obj.__dict__.items():
            if isinstance(attr_val, ScopeSpec):

                for var_name, var in attr_val._variables.items():

                    if var.id == resolve_var.id:
                        return True, [attr_name], var_name
            if isinstance(attr_val, ItemsSpec):
                found, path_, var_name = resolve_variable(attr_val, resolve_var)
                if found:
                    return found, [attr_name] + path_, var_name

    elif isinstance(obj, ItemsSpec):
        for item_name, item in obj._items.items():
            found, path_, var_name = resolve_variable(item, resolve_var)
            if found:
                return found, [item_name] + path_, var_name
    elif isinstance(obj, ModuleSpec):

        for attr_name, attr_val in obj._namespaces.items():
            if isinstance(attr_val, ScopeSpec):

                for var_name, var in attr_val._variables.items():
                    if var.id == resolve_var.id:

                        return True, [attr_name], var_name
        for item_name, item in obj._item_specs.items():

            found, path_, var_name = resolve_variable(item, resolve_var)
            if found:
                return found, [item_name] + path_, var_name
    else:
        raise TypeError('No compatible object '+str(obj))

    return False, [], ""

def recursive_get_attr(obj, attr_list):

    attr_ = getattr(obj, attr_list[0])

    if len(attr_list)>1:
        return recursive_get_attr(attr_, attr_list[1:])
    else:
        return attr_

#class AssignTypes(Enum):

#    ADD = 0
#    ASSIGN = 1

#def assign(a, b):
#    return a, b, AssignTypes.ASSIGN

#def add(a, b)
class MappingFailed(Exception):
    ...

class Module(Subsystem, Item, EquationBase):

    tag = None

    def __new__(cls, *args, **kwargs):

        obj = object.__new__(cls)
        obj._scopes = {}
        obj._item_specs = {}
        obj._resolved_mappings = []
        if cls.tag is not None:
            obj.tag = cls.tag

        if obj.tag is None:
            obj.tag = cls.__name__.lower()

        super(Module, obj).__init__(obj.tag)

        parent_subsystem = _active_subsystem.get_active_manager_context(ignore_no_context=True)

        if not hasattr(cls, '__org_init__'):
            cls.__org_init__ = cls.__init__

        def wrap(*args, **kwargs):
            _active_subsystem.clear_active_manager_context(parent_subsystem)
            _active_subsystem.set_active_manager_context(obj)
            cls.__org_init__(*args, **kwargs)
            _active_subsystem.clear_active_manager_context(obj)
            _active_subsystem.set_active_manager_context(parent_subsystem)

            scope_specs = {}
            bases = cls.__bases__
            for b in bases:
                scope_specs.update(b.__dict__)

            scope_specs.update(cls.__dict__)


            for b in cls.__bases__:
                for k, v in b.__dict__.items():

                    if issubclass(v.__class__, ScopeSpec):
                        if k in scope_specs:
                            bases = scope_specs[k].__class__.__bases__
                            if v.__class__ == scope_specs[k].__class__ or v.__class__ in bases:
                                scope_specs[k]._equations += v._equations
                        else:
                            scope_specs[k] = v

                    #if issubclass(v.__class__, ModuleMappings):
                    #    v._class = b
                    #    if k in scope_specs:

                    #        scope_specs[k+b.__name__] = v
                    #        pass
                    #    else:
                    #        scope_specs[k] = v



            for k, v in scope_specs.items():


                if isinstance(v, ScopeSpec):
                    cls.add_scope(obj, k, v, kwargs)

                elif isinstance(v, ItemsSpec):
                    obj._item_specs[k] = v

                #    v.check_assigned()

                elif isinstance(v, ModuleMappings):
                    obj_ =  obj
                    for mapping_ in v.mappings:

                        mapto = mapping_[0]
                        mapfrom = mapping_[1]
                        map_type = mapping_[2]
                        try:
                            resolved_to = resolve_variable(obj_, mapto)
                            resolved_from = resolve_variable(obj_, mapfrom)
                            obj._resolved_mappings.append((resolved_to, resolved_from))



                            if map_type == MappingTypes.ASSIGN:

                                try:
                                    to_attr = recursive_get_attr(obj_, resolved_to[1])
                                except AttributeError as ae:
                                    raise MappingFailed(f"!")

                                try:
                                    from_attr = getattr(recursive_get_attr(obj_, resolved_from[1]), resolved_from[2])
                                except AttributeError as ae:
                                    raise MappingFailed(f"The obj_ect {obj_}, {resolved_from[1]} has no variable {resolved_from[2]}")
                                except IndexError:
                                    raise MappingFailed(f"The obj_ect {obj_}, {resolved_from[1]} has no variable {resolved_from[2]}")


                                setattr(to_attr, resolved_to[2], from_attr)
                            elif map_type == MappingTypes.ADD:
                                try:
                                    to_attr = recursive_get_attr(obj_, resolved_to[1] + [resolved_to[2]])
                                except AttributeError as ae:
                                    raise MappingFailed(f"The namespace {'.'.join([str(s) for s in [obj_]+resolved_to[1]])} does not have a variable {resolved_to[2]}.")

                                try:
                                    from_attr = getattr(recursive_get_attr(obj_, resolved_from[1]), resolved_from[2])
                                except AttributeError as ae:
                                    raise MappingFailed(f"The obj_ect {obj_}, {resolved_from[1]} has no variable {resolved_from[2]}")

                                to_attr.__iadd__(from_attr)

                            else:
                                raise ValueError('Unknown mapping type: ', map_type)


                        except:
                            raise



            if isinstance(parent_subsystem, Subsystem):
                parent_subsystem.register_item(obj)

        cls.__init__ = wrap

        return obj

    def __init__(self, tag=None):#, **kwargs):
        if tag is not None:
            self.tag = tag
        #for k, v in kwargs.items():
        #    if hasattr(self, k) and isinstance(getattr(self,k), ScopeSpec):
        #        for var, val in v.items():
        #            getattr(getattr(self,k), var).value = val

    def _finalize(self):

        for itemspec_name, itemspec in self._item_specs.items():
            print(itemspec_name)
            itemspec.finalize()

        for item_name, item in self.registered_items.items():
            print(item_name)
            if isinstance(item, Module):
                item.finalize()


    def prefinalize(self):
        ...

    def finalize(self):

        self.prefinalize()
        self._finalize()

    @classmethod
    def add_scope(cls, obj, namespace, scope, values:dict):
        ns = obj.create_namespace(namespace)
        obj._scopes[namespace] = scope
        #setattr(obj, namespace, ns)

        class Eq(EquationBase):

            tag = namespace + '_eq'

        eq_ = Eq()

        for eq_func in scope._equations:
            eq_func.__self__ = obj
            eq_func_dec = add_equation(eq_, eq_func)

            #print(eq_func.__name__)
            #setattr(obj, eq_func.__name__, types.MethodType(eq_func_dec, obj))



        prescope = list(scope.__dict__.items())

        for v_name, v_ in prescope:

            if isinstance(v_, Variable):

                if v_name in values:
                    val = values.pop(v_name)

                else:
                    val = v_.value

                if v_._vartype == 'State':

                    eq_.add_state(v_name, val)
                else:
                    if v_.construct:
                        if v_._vartype == "parameter":
                            eq_.add_parameter(v_name, val, var_type=var_type_map[v_._vartype], logger_level=None, alias=None, integrate=v_.integrate)

                        elif v_._vartype != "deriv":

                            eq_.add_variable(v_name, val, var_type=var_type_map[v_._vartype], logger_level=None, alias=None)

        scope.generate_variables(eq_)
        ns.add_equations([eq_])

    @classmethod
    def clone_tree(cls):
        clone = ModuleSpec(cls)
        return clone

    def get_path(self):
        return ".".join(self.path)

class MappingTypes(Enum):

    ASSIGN = 0
    ADD = 1

class ModuleMappings:

    mappings = []

    def __init__(self, *args):
        self.mappings = list(args)

    def __enter__(self):
        _active_mappings.set_active_manager_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _active_mappings.clear_active_manager_context(self)

    def assign(self, a, b):
        self.mappings.append((a, b, MappingTypes.ASSIGN))


    def add(self, a, b):
        self.mappings.append((a, b, MappingTypes.ADD))

def create_mappings(*args):

    return ModuleMappings(*args)


class ScopeSpec:

    def __init__(self, parent_path=[], new_var_ids=False):

        self._variables = {}

        self._equations = []

        class_var = {}

        for b in self.__class__.__bases__:
            class_var.update(b.__dict__)

        class_var.update(self.__class__.__dict__)

        for k, v in class_var.items():

            if isinstance(v, Variable):
                #instance = v.instance(id=".".join(parent_path+[k]), name=k)
                instance = v.instance(id=str(uuid4()) if new_var_ids else v.id, name=k)
                setattr(self, k, instance)
                
                self._variables[k] = instance
    
    def __setattr__(self, key, value, clone=False):

        if clone:
            super(ScopeSpec, self).__setattr__(key, value)
            return
        elif isinstance(value, PartialResult) and not isinstance(value, Variable):
            if value.op == Operations.ADD and value.arguments[0] == getattr(self, key) and isinstance(value.arguments[1], Variable):
                mapping_manager = _active_mappings.get_active_manager_context()
                mapping_manager.add(value.arguments[0], value.arguments[1])
            else:
                raise ValueError('Partial result not matching additive mapping.')
        elif hasattr(self, '_variables') and key in self._variables:
            mapping_context: ModuleMappings = _active_mappings.get_active_manager_context()
            map_to = getattr(self, key)
            mapping_context.assign(map_to, value)

        else:
            super(ScopeSpec, self).__setattr__(key, value)

    #def clone(self, parent_path=[]):
    #    return self.__class__(parent_path)

    def _instance(self, parent_path=[]):
        return self.__class__(parent_path)

    def clone(self):
        clone = self.__class__(new_var_ids=True)
        #var_list = list(clone.__dict__.items())
        #for var, val in var_list:
        #    if isinstance(val, Variable):
        #        clone.__setattr__(var, val.clone(id=str(uuid4()), name=val.name), clone=True)

        return clone

    def generate_variables(self, equation):
        ...
""" 
def add_edges(node, graph: nx.DiGraph):
    graph.add_edge(node.a, node)
    if not isinstance(node.a, Variable):
        add_edges(node.a, graph)

    graph.add_edge(node.b, node)
    if not isinstance(node.b, Variable):
        add_edges(node.b, graph)"""

import inspect
def allow_implicit(func):
    sign = inspect.signature(func)


    def check_implicit(*args, **kwargs):

        if len(args)>1 and isinstance(item:=args[1], Item):
            _kwargs = {}
            _args = []
            for n, p in sign.parameters.items():
                if n != "self":
                    val = getattr(getattr(item, 'variables'),n)
                    if n in kwargs:

                        _kwargs[n] = val
                    else:
                        _args.append(val)


            func(args[0], *_args, **kwargs)
        else:
            func(*args, **kwargs)


    return check_implicit


from numerous.multiphysics.equation_decorators import Equation

class EquationSpec(object):

    def __init__(self, scope):
        self.scope: ScopeSpec = scope

    def __call__(self, func):

        self.scope._equations.append(func)



        return func