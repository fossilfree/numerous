from abc import ABC, abstractmethod

class ScopeSpecInterface(ABC):
    ...

class ModuleSpecInterface(ABC):

    @abstractmethod
    def from_module_cls(self, annotations):
        ...

class ItemsSpecInterface(ABC):
    ...

class ModuleInterface(ABC):
    ...

class ConnectorInterface(ABC):
    channels: dict

    @abstractmethod
    def instance(self, context:dict):
        ...

class ModuleConnectionsInterface(ABC):
    ...

class EquationSpecInterface(ABC):
    ...

class VariableInterface(ABC):

    @abstractmethod
    def instance(self, context:dict):
        ...