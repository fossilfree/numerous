from numerous.declarative.interfaces import ModuleInterface, ItemsSpecInterface, ModuleSpecInterface
from numerous.declarative.instance import Class, get_class_vars
from .module import handler
from .exceptions import ItemNotAssignedError
class ItemsSpec(Class, ItemsSpecInterface):
    _modules: dict

    def __init__(self, init=True):
        super(ItemsSpec, self).__init__()
        if init:
            modules = get_class_vars(self, (ModuleInterface,), _handle_annotations=handler)
            print(modules)
            context = {}

            self._modules = {}
            for k, v in modules.items():
                instance = v.instance(context)

                self._modules[k] = instance
                setattr(self, k, instance)

    def _instance_recursive(self, context):

        instance = ItemsSpec(init=False)
        instance._modules = {k: m.instance(context) for k, m in self._modules.items()}

        for name, module in instance._modules.items():

            setattr(instance, name, module)

        return instance

    def get_modules(self, include_specs=False, ignore_not_assigned=False):
        if include_specs:
            module_specs = self._items_of_type(ModuleSpecInterface)

            items = {}
            not_assigned = {}

            for name, module_spec in module_specs.items():
                if module_spec.assigned_to:
                    items[name] = module_spec.assigned_to
                else:
                    not_assigned[name] = module_spec

            if not ignore_not_assigned and len(not_assigned) > 0:
                raise ItemNotAssignedError(f"Items not assigned: {list(not_assigned.keys())}")

            items.update(self._items_of_type(ModuleInterface))
        else:
            return self._items_of_type(ModuleInterface)

        return items

    @property
    def modules(self):

        return self.get_modules(include_specs=True)

    def __setattr__(self, key, value):
        if isinstance(value, ModuleInterface):
            # TODO add checks on if assigned etc
            value.module_spec = getattr(self, key)

            self._modules[key].assigned_to = value

            #check mappings and connections



        super(ItemsSpec, self).__setattr__(key, value)
