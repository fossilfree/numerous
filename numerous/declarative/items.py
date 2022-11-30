from .interfaces import ItemsSpecInterface, ModuleSpecInterface, ModuleInterface
from .clonable_interfaces import ClassVarSpec
from .module import ModuleSpec

class ItemsSpec(ItemsSpecInterface, ClassVarSpec):

    def __init__(self):

        super(ItemsSpec, self).__init__(class_var_type=ModuleInterface, handle_annotations=ModuleSpec.from_module)


    def get_items(self):
        return self._references





