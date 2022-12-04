from .interfaces import ItemsSpecInterface, ModuleSpecInterface, ModuleInterface
from .clonable_interfaces import ClassVarSpec, ParentReference
from .module import ModuleSpec, Module
from .exceptions import ItemNotAssignedError, ItemAlreadyAssigned, ItemUndeclaredWarning
import warnings


class ItemsSpec(ItemsSpecInterface, ClassVarSpec):

    def __init__(self):

        super(ItemsSpec, self).__init__(class_var_type=ModuleInterface, handle_annotations=ModuleSpec.from_module)

    def get_modules(self, check=True):

        modules_and_specs = self.get_references_of_type((ModuleSpecInterface, ModuleInterface))

        modules_self = {}

        for k, v in modules_and_specs.items():
            module = getattr(self, k)

            if v!= module:
                modules_self[k+"_spec"] = v

            if isinstance(module, Module):
                ...
            elif module._assigned_to is not None:
                module = module.assigned_to
            elif isinstance(module, ModuleSpec) and check:
                raise ItemNotAssignedError(f"Found an item which has not been assigned to a module <{k}>.")
            modules_self[k] = module

        return modules_self


    def __setattr__(self, key, value, add_ref=False):
        if hasattr(self, key):
            if isinstance(assigned_item:=getattr(self, key), (ModuleSpec,)) \
                    and isinstance(value, (ModuleSpec, Module)):

                assigned_item.assigned_to = value
                if not value._parent:
                    value.set_parent(ParentReference(self, key))
            elif isinstance(getattr(self, key), (Module,)):
                raise ItemAlreadyAssigned("The module has already been assigned.")
        else:
            if isinstance(value, (ModuleSpec, Module)) and not add_ref:

                if not value._parent:

                    self.add_reference(key, value)
                    value.set_parent(ParentReference(self, key))

                    warnings.warn(f"Item {value._parent.attr} added dynamically.", ItemUndeclaredWarning)


        super(ItemsSpec, self).__setattr__(key, value)
            



