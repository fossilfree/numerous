from __future__ import annotations

from enum import Enum

from numerous.declarative.context_managers import _active_mappings
from numerous.declarative.variables import Variable


class Obj:
    def __init__(self, scope_specs, items_specs):
        self._item_specs = items_specs
        self._scopes = scope_specs


class MappingTypes(Enum):

    ASSIGN = 0
    ADD = 1


class ModuleMappings:

    mappings = []

    def __init__(self, *args):

        super(ModuleMappings, self).__init__()

        self.mappings = list(args)

    def __enter__(self):
        _active_mappings.set_active_manager_context(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _active_mappings.clear_active_manager_context(self)

    def assign(self, a, b):
        if not isinstance(a, Variable):
            raise TypeError(f"a is not a Variable, but {a.__class__.__name__}")
        if not isinstance(b, Variable):
            raise TypeError(f"b is not a Variable, but {b.__class__.__name__}")
        self.mappings.append((a, b, MappingTypes.ASSIGN))
        a.mapped_to.append(b)


    def add(self, a, b):
        self.mappings.append((a, b, MappingTypes.ADD))
        a.mapped_to.append(b)


def create_mappings(*args):

    return ModuleMappings(*args)
