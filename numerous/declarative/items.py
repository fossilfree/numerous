from numerous.declarative.interfaces import ModuleInterface, ItemsSpecInterface, ModuleSpecInterface
from numerous.declarative.instance import Class, get_class_vars
from .module import handler
from .exceptions import ItemNotAssignedError
class ItemsSpec(Class, ItemsSpecInterface):
    _modules: dict
    _module_specs: dict

    def __init__(self, init=True):
        super(ItemsSpec, self).__init__()
        if init:
            module_specs = get_class_vars(self, (ModuleInterface,), _handle_annotations=handler)

            context = {}

            self._modules = {}
            self._module_specs = {}

            for k, v in module_specs.items():
                instance = v.instance(context)

                self._module_specs[k] = instance
                setattr(self, k, instance)


    def _instance_recursive(self, context):

        instance = ItemsSpec(init=False)
        instance._modules = {k: m.instance(context) for k, m in self._modules.items()}
        instance._module_specs = {k: m.instance(context) for k, m in self._module_specs.items()}

        for name, module in instance._module_specs.items():

            setattr(instance, name, module)

        for name, module in instance._modules.items():

            setattr(instance, name, module)

        return instance

    def update_modules(self):

        for k, v in self._module_specs.items():
            if k not in self._modules and v.assigned_to:
                self._modules[k] = v.assigned_to

    def get_modules(self, ignore_not_assigned=False, update_first=False):
        if update_first:
            self.update_modules()

        if not ignore_not_assigned:
            not_assigned = {}
            for k, v in self._module_specs.items():
                if k not in self._modules:

                    not_assigned[k] = v

            if len(not_assigned) > 0:
                for v in not_assigned.values():
                    print(v.assigned_to)
                    print(v.path)
                raise ItemNotAssignedError(f"Items not assigned: {list(not_assigned.keys())}")

        return self._modules

    def get_module_specs(self):

        return self._module_specs


    @property
    def modules(self):

        return self._modules

    def __setattr__(self, key, value):
        if isinstance(value, ModuleInterface):
            # TODO add checks on if assigned etc
            if not key in self._module_specs:
                raise ModuleNotFoundError(f"Items spec does not have a module {key}")
            self._module_specs[key].assigned_to = value
            self._modules[key] = value
            value._parent = True
            #check mappings and connections



        super(ItemsSpec, self).__setattr__(key, value)
