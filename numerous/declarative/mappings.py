from __future__ import annotations

import uuid
from enum import Enum

from .interfaces import ModuleInterface
from .context_managers import _active_mappings
from .variables import Variable, MappingTypes


class Obj:
    def __init__(self, scope_specs, items_specs):
        self._item_specs = items_specs
        self._scopes = scope_specs


class Mapping:
    _mapping_type: MappingTypes
    _from: Variable
    _to: Variable
    def __init__(self, to_var:Variable=None, from_var:Variable=None, mapping_type:MappingTypes=None):
        super(Mapping, self).__init__()

        self._mapping_type = mapping_type
        self._to = to_var
        self._from = from_var

    @property
    def mapping_type(self):
        return self._mapping_type

class ModuleMappings():
    _mappings: list
    def __init__(self, *args):

        super(ModuleMappings, self).__init__()
        self._mappings = []

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
        self._mappings.append(Mapping(a, b, MappingTypes.ASSIGN))
        #a.mapped_to.append(b)

    def add(self, a, b):
        self._mappings.append(Mapping(a, b, MappingTypes.ADD))

    @property
    def mappings(self):
        return self._mappings


#def create_mappings(*args):
    #return ModuleMappings(*args)