from .interfaces import ScopeSpecInterface
from .clonable_interfaces import ClassVarSpec
from .variables import Variable, Operations
from .context_managers import _active_mappings, _active_subsystem, NoManagerContextActiveException
from .exceptions import MappingOutsideMappingContextError, ItemNotAssignedError, NoSuchItemInSpec, MappingFailed, \
    NotMappedError, FixedMappedError
from .mappings import Obj, MappingTypes, ModuleMappings

class ScopeSpec(ScopeSpecInterface, ClassVarSpec):

    _equations: list
    _initialized: bool = False

    def __init__(self):

        self._equations = []


        super(ScopeSpec, self).__init__(class_var_type=Variable)


    def get_variables(self):
        return self._references

    @property
    def equations(self):
        return self._equations

    def clone(self):
        clone = super(ScopeSpec, self).clone()
        clone._equations = self._equations
        return clone

    def __setattr__(self, key, value, add_ref=False):

        if self._initialized and isinstance(value, Variable) and hasattr(self, key) and getattr(self, key) != value:

            getattr(self, key).assign(value)
        else:
            super(ScopeSpec, self).__setattr__(key, value)

    def set_values(self, **kwargs):

        variables = self.get_variables()

        for k, v in kwargs.items():
            variables[k].value = v
