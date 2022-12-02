from .interfaces import ModuleSpecInterface, ScopeSpecInterface, ItemsSpecInterface, ModuleInterface, ConnectorInterface
from .bus import BusInterface
from .clonable_interfaces import ClassVarSpec, get_class_vars
from .variables import Variable


class ModuleSpec(ModuleSpecInterface, ClassVarSpec):
    _assigned_to: ModuleInterface = None

    def __init__(self):

        super(ModuleSpec, self).__init__(clone_refs=True)

    @classmethod
    def from_module(cls, module:ModuleInterface):
        module_vars = get_class_vars(module, (ScopeSpecInterface, ItemsSpecInterface, ConnectorInterface))
        module_spec = cls()
        module_spec.configure_clone(module_spec, module_vars, do_clone=True)
        return module_spec

    def clone(self):
        clone = super(ModuleSpec, self).clone()
        clone._assigned_to = self._assigned_to

        return clone

    def get_connectors(self):

        return self.get_references_of_type(ConnectorInterface)

    def get_scope_specs(self):

        return self.get_references_of_type(ScopeSpecInterface)

    def get_items_specs(self):

        return self.get_references_of_type(ItemsSpecInterface)

class Module(ModuleInterface, ClassVarSpec):
    def __init__(self):

        super(Module, self).__init__(class_var_type=(ScopeSpecInterface, ItemsSpecInterface, ConnectorInterface))

    def get_scope_specs(self):

        return self.get_references_of_type(ScopeSpecInterface)

    def get_items_specs(self):

        return self.get_references_of_type(ItemsSpecInterface)

    def get_connectors(self):

        return self.get_references_of_type(ConnectorInterface)
