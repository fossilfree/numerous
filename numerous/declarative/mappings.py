from __future__ import annotations

import uuid
from enum import Enum
from .clonable_interfaces import Clonable
from .interfaces import ModuleInterface
from .context_managers import _active_mappings
from .variables import Variable, MappingTypes


class Obj:
    def __init__(self, scope_specs, items_specs):
        self._item_specs = items_specs
        self._scopes = scope_specs


class Mapping(Clonable):
    _mapping_type: MappingTypes
    def __init__(self, to_var:ModuleInterface=None, from_var:ModuleInterface=None, mapping_type:MappingTypes=None):
        super(Mapping, self).__init__(set_parent_on_references=False, bind=True)

        self._mapping_type = mapping_type
        if to_var:
            self.add_reference('to_var', to_var)
        if from_var:
            self.add_reference('from_var', from_var)
    def clone(self):
        clone = super(Mapping, self).clone()
        clone._mapping_type = self._mapping_type
        return clone

    @property
    def mapping_type(self):
        return self._mapping_type

class ModuleMappings(Clonable):

    def __init__(self, *args):

        super(ModuleMappings, self).__init__(set_parent_on_references=False, bind=False)

    def __enter__(self):
        _active_mappings.set_active_manager_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _active_mappings.clear_active_manager_context(self, ignore_not_set=True)

    def assign(self, a, b):
        if not isinstance(a, Variable):
            raise TypeError(f"to_var is not a Variable, but {a.__class__.__name__}")
        if not isinstance(b, Variable):
            raise TypeError(f"from_var is not a Variable, but {b.__class__.__name__}")
        self.add_reference(str(uuid.uuid4()), Mapping(a, b, MappingTypes.ASSIGN))
        #a.mapped_to.append(b)

    def add(self, a, b):
        self.add_reference(str(uuid.uuid4()), Mapping(a, b, MappingTypes.ADD))

    @property
    def mappings(self):
        return self.get_references_of_type((Mapping,))


def create_mappings(*args):
    return ModuleMappings(*args)
