from .interfaces import ModuleSpecInterface, ScopeSpecInterface, ItemsSpecInterface, ModuleInterface, ConnectorInterface
from .bus import BusInterface
from .clonable_interfaces import ClassVarSpec, get_class_vars
from .variables import Variable


class ModuleSpec(ModuleSpecInterface, ClassVarSpec):
    
    def __init__(self):

        super(ModuleSpec, self).__init__(clone_refs=True)

    @classmethod
    def from_module(cls, module:ModuleInterface):
        module_vars = get_class_vars(module, (ScopeSpecInterface, ItemsSpecInterface, ConnectorInterface))
        module_spec = cls()
        module_spec.configure_clone(module_spec, module_vars, do_clone=True)
        return module_spec



class Module(ModuleInterface, ClassVarSpec):

    def __init__(self):

        super(Module, self).__init__(class_var_type=(ScopeSpecInterface, ItemsSpecInterface, ConnectorInterface))