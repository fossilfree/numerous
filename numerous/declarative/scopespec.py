from .instance import Class, get_class_vars
from .interfaces import ScopeSpecInterface, VariableInterface

class ScopeSpec(Class, ScopeSpecInterface):
    _equations: list
    _variables: dict

    def __init__(self, context=None, init=True):
        super(ScopeSpec, self).__init__()

        if init:
            self._equations = []

            context = {} if context is None else context
            self._variables = {k: v.instance(context) for k, v in get_class_vars(self, (VariableInterface,)).items()}

            for name, variable in self._variables.items():
                setattr(self, name, variable)

    def _instance_recursive(self, context):
        instance = self.__class__(init=False)
        instance._equations = self._equations
        instance._variables = {k: v.instance(context) for k, v in self._variables.items()}
        for name, variable in instance._variables.items():
            setattr(instance, name, variable)

        return instance

    @property
    def equations(self):
        return self._equations

    @property
    def variables(self):
        return self._variables
