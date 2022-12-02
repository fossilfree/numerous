from .interfaces import ItemsSpecInterface, ModuleSpecInterface, ModuleInterface
from .clonable_interfaces import ClassVarSpec
from .module import ModuleSpec, Module
from .exceptions import ItemNotAssignedError, ItemAlreadyAssigned
class ItemsSpec(ItemsSpecInterface, ClassVarSpec):

    def __init__(self):

        super(ItemsSpec, self).__init__(class_var_type=ModuleInterface, handle_annotations=ModuleSpec.from_module)

    def get_modules(self, check=True):

        modules_and_specs = self.get_references_of_type((ModuleSpec, Module))

        modules_self = {}

        for k, v in modules_and_specs.items():
            module = getattr(self, k)
            if isinstance(module, Module):
                ...
            elif module._assigned_to is not None:
                module = module.assigned_to
            elif isinstance(module, ModuleSpec) and check:
                raise ItemNotAssignedError(f"Found an item which has not been assigned to a module <{k}>.")
            modules_self[k] = module

        return modules_self


    def __setattr__(self, key, value):
        if hasattr(self, key):
            if isinstance(assigned_item:=getattr(self, key), (ModuleSpec,)) \
                    and isinstance(set_item:=value, (ModuleSpec, Module)):

                assigned_item.assigned_to = set_item
            elif isinstance(getattr(self, key), (Module,)):
                raise ItemAlreadyAssigned("The module has already been assigned.")

        super(ItemsSpec, self).__setattr__(key, value)
            



