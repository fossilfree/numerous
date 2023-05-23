import uuid
from .interfaces import ModuleSpecInterface, ModuleInterface, ItemsSpecInterface, ConnectorInterface, ModuleConnectionsInterface, ScopeSpecInterface
from .instance import get_class_vars, Class
from .context_managers import _active_subsystem, NoManagerContextActiveException
from .connector import ModuleConnections

class AutoItemsSpec(Class, ItemsSpecInterface):
    _modules: dict
    def __init__(self, modules:list):
        super(AutoItemsSpec, self).__init__()
        counter = 0
        for module in modules:
            if not hasattr(module, 'tag') or not module.tag:
                module.tag = "unnamed_"+str(counter)
                counter+=1
        self._modules = {module.tag: module for module in modules}
        setattr(self, module.tag, module)

    def get_modules(self, ignore_not_assigned=False, update_first=False):

        return self._modules

    def get_module_specs(self):

        return {}

    @property
    def modules(self):
        return self._modules

    def remove_non_orphants(self):

        orphants = {}

        for name, module in self.modules.items():
            if module._parent is None:
                orphants[name] = module
                setattr(self, name, module)
        self.modules = orphants


def local(tag, mod: ModuleInterface):
    mod.tag = tag

    return mod

class ModuleSpec(Class, ModuleSpecInterface):
    assigned_to: ModuleInterface|None = None
    path: tuple[str] | None
    def __init__(self, items:dict):
        super(ModuleSpec, self).__init__()
        self._items = items
        self.path = None

        for k, v in self._items.items():
            setattr(self, k, v)

    @classmethod
    def from_module_cls(cls, module):
        items = get_class_vars(module, (Class,))
        context = {}
        module_spec = cls({k: v.instance(context) for k, v in items.items()})
        return module_spec

    def _instance_recursive(self, context:dict):
        instance = ModuleSpec({k: v.instance(context) for k, v in self._items.items()})

        return instance

    @property
    def scopes(self):
        return self._items_of_type(ScopeSpecInterface)

    @property
    def items_specs(self):
        return self._items_of_type(ItemsSpecInterface)

class RegisterHelper:

    def __init__(self):
        self._items = {}

    def register_item(self, item):
        #if item.tag in self._items:
        #    raise DuplicateItemError(f"An item with tag {item.tag} already registered.")
        self._items[str(uuid.uuid4())] = item

    def get_items(self):
        return self._items


class Module(ModuleInterface, Class):
    module_spec: ModuleSpec|None = None
    _parent = None
    path: tuple[str]|None
    def __new__(cls, *args, **kwargs):

        parent_module = _active_subsystem.get_active_manager_context(ignore_no_context=True)
        register_helper = RegisterHelper()

        _active_subsystem.clear_active_manager_context(parent_module)
        _active_subsystem.set_active_manager_context(register_helper)

        org_init = cls.__init__

        def wrap(self, *args, **kwargs):

            org_init(self, *args, **kwargs)
            _active_subsystem.clear_active_manager_context(register_helper)
            _active_subsystem.set_active_manager_context(parent_module)

            if isinstance(parent_module, RegisterHelper):

                parent_module.register_item(self)

            _auto_modules = []
            for module in register_helper.get_items().values():
                if module._parent is None:
                    _auto_modules.append(module)
                    module._parent = self
            if len(_auto_modules)>0:
                self._auto_modules = AutoItemsSpec(_auto_modules)

            cls.__init__ = org_init

        cls.__init__ = wrap

        instance = Class.__new__(cls)

        return instance

    def __init__(self):
        super(Module, self).__init__()

        self.path = None
        vars = get_class_vars(self, (Class,))

        context = {}

        self._items = {}

        for k, v in vars.items():
            instance =  v.instance(context)
            setattr(self, k, instance)
            self._items[k] = instance

    @property
    def connectors(self):
        return self._items_of_type(ConnectorInterface)

    @property
    def items_specs(self):
        return self._items_of_type(ItemsSpecInterface)

    @property
    def connection_sets(self):
        #connection_sets = {k: v for k, v in self.__class__.__dict__.items() if isinstance(v, ModuleConnections)}
        connection_sets = {k: v for k, v in self.__dict__.items() if isinstance(v, ModuleConnections)}

        return connection_sets

    @property
    def scopes(self):
        return self._items_of_type(ScopeSpecInterface)

    def finalize(self):
        ...

def handler(annotation):
    return ModuleSpec.from_module_cls(annotation)



class BoundValues:

    def __init__(self, **kwargs):

        self.bound_values = kwargs
