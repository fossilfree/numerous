from abc import abstractmethod, ABC


class NumbaCallbackBase(ABC):

    def __init__(self):
        self.numba_params_spec = {}

    def register_numba_varaible(self, variable_name, numba_variable_type):
        self.numba_params_spec.update({variable_name: numba_variable_type})

    def restore_variables_from_numba(self,numba_model,var_list):
        for variables_name in self.numba_params_spec.keys():
            self.__setattr__(variables_name,getattr(numba_model, variables_name))
        self.finalize(var_list)

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def finalize(self):
        pass