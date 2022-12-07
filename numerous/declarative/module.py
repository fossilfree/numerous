import uuid

from .interfaces import ModuleSpecInterface, ScopeSpecInterface, ItemsSpecInterface, ModuleInterface, ConnectorInterface, ModuleConnectionsInterface
from .mappings import ModuleMappings, create_mappings
from numerous.declarative.context_managers import _active_mappings
from .clonable_interfaces import ClassVarSpec, get_class_vars, Clonable, ParentReference
from .variables import Variable
from .utils import recursive_get_attr, RegisterHelper

from .context_managers import _active_subsystem, NoManagerContextActiveException

class ModuleSpec(ModuleSpecInterface, ClassVarSpec):
    _assigned_to: ModuleInterface = None

    def __init__(self):

        super(ModuleSpec, self).__init__(clone_refs=True)

    @classmethod
    def from_module(cls, module:ModuleInterface):
        module_vars = get_class_vars(module, (ScopeSpecInterface, ItemsSpecInterface, ModuleConnectionsInterface, ConnectorInterface, ModuleMappings))
        module_spec = cls()
        module_spec.configure_clone(module_spec, module_vars, do_clone=True)
        return module_spec

    def get_path(self, host):
        if self._assigned_to:
            return self._assigned_to.get_path(host)
        else:
            return super(ModuleSpec, self).get_path(host)
    
    def clone(self):
        clone = super(ModuleSpec, self).clone()
        clone._assigned_to = self._assigned_to

        return clone

    def get_connection_sets(self):

        return self.get_references_of_type(ModuleConnectionsInterface)

    def get_scope_specs(self):

        return self.get_references_of_type(ScopeSpecInterface)

    def get_items_specs(self):

        return self.get_references_of_type(ItemsSpecInterface)

    @property
    def assigned_to(self):
        return self._assigned_to

    @assigned_to.setter
    def assigned_to(self, val):
        self._assigned_to = val

class AutoItemsSpec(Clonable, ItemsSpecInterface):

    def __init__(self, modules:list):
        counter = 0
        for module in modules:
            if not module.tag:
                module.tag = "unnamed_"+str(counter)
                counter+=1
        self.modules = {module.tag: module for module in modules}
    def get_modules(self, check=False):
        return self.modules

    def remove_non_orphants(self):

        orphants = {}

        for name, module in self.modules.items():
            if module._parent is None:
                orphants[name] = module
                module.set_parent(ParentReference(self, name))
                setattr(self, name, module)
        self.modules = orphants


def local(tag, mod: ModuleInterface):
    mod.tag = tag

    return mod

class Module(ModuleInterface, ClassVarSpec):

    _initialized = False
    _processed = False

    tag: None

    def __new__(cls, *args, **kwargs):
        parent_module = _active_subsystem.get_active_manager_context(ignore_no_context=True)
        register_helper = RegisterHelper()

        _active_subsystem.clear_active_manager_context(parent_module)
        _active_subsystem.set_active_manager_context(register_helper)

        org_init = cls.__init__

        def wrap(self, *args, **kwargs):

            #Call original init

            parent_mappings = _active_mappings.get_active_manager_context(ignore_no_context=True)
            _active_mappings.clear_active_manager_context(parent_mappings)

            with create_mappings() as internal_mappings:
                org_init(self, *args, **kwargs)

            _active_mappings.set_active_manager_context(parent_mappings)

            self.add_reference("internal_mappings", internal_mappings)

            cls.__init__ = org_init
            _active_subsystem.clear_active_manager_context(register_helper)
            _active_subsystem.set_active_manager_context(parent_module)

            if isinstance(parent_module, RegisterHelper):
                parent_module.register_item(self)

            _auto_modules = []
            for module in register_helper.get_items().values():
                if module._parent is None:
                    _auto_modules.append(module)
            self._auto_modules = AutoItemsSpec(_auto_modules)
            #if len(_auto_modules)>0:

            self.add_reference('unbound', self._auto_modules)

        cls.__init__ = wrap

        instance = object.__new__(cls)

        return instance

    def __init__(self):
        self.tag = None
        _class_var_type = (ScopeSpecInterface, ItemsSpecInterface, ModuleConnectionsInterface, ConnectorInterface, ModuleMappings)



        if not self.__class__._initialized:
            get_class_vars(self.__class__, _class_var_type)
            self.__class__._initialized = True

        super(Module, self).__init__(class_var_type=_class_var_type)

    def set_tag(self, tag):
        self.tag = tag

    def get_scope_specs(self):

        return self.get_references_of_type(ScopeSpecInterface)

    def get_items_specs(self):
        items_specs = self.get_references_of_type(ItemsSpecInterface)
        return items_specs

    def get_connection_sets(self):

        return self.get_references_of_type(ModuleConnectionsInterface)

    def get_mappings(self):

        return self.get_references_of_type(ModuleMappings)

    @classmethod
    def merge(cls, current: dict, update: dict, types: tuple):
        #return super(Module, cls).merge(current, update)
        for k, v in update.items():
            if isinstance(v, ScopeSpecInterface):
                if k in current:
                    for e in current[k].equations:
                        if e not in v.equations:
                            v.equations.append(e)
                current[k] = v

            elif isinstance(v, types):
                current[k] = v



        return current
class BoundValues:

    def __init__(self, **kwargs):

        self.bound_values = kwargs
