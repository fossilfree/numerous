import uuid
from collections import deque

from numerous.engine.variables import VariableBase, VariableDescription, VariableType


class VariableDescriptionMap(VariableBase):
    def __init__(self, equation):
        self.variables_descriptions = super().__dict__
        self.equation = equation
        self.variables_descriptions_deque = deque()

    def register_variable_description(self, variable_description):
        if variable_description.tag not in self.variables_descriptions:
            self.variables_descriptions[variable_description.tag] = variable_description
        else:
            raise ValueError('Variable description with tag {} is already exist in equation {}'.format(
                variable_description.tag, self.equation.tag))

    # refactored in a more functional way
    def __iter__(self):
        # used filter and extend
        self.variables_descriptions_deque.extend(list(filter((lambda v:\
            isinstance(v, VariableDescription)),\
            self.variables_descriptions.values())))
        return self

    def __next__(self):
        if self.variables_descriptions_deque:
            return self.variables_descriptions_deque.pop()
        else:
            raise StopIteration


class EquationBase:
    """
    Equations governing the behavior of the objects in the model and their interactions
     are defined in classes extending the :class:`numerous.multiphysics.Equation`.

    """
    def __init__(self, tag=None):
        if tag:
            self.tag = tag
        self.equations = []
        self.variables_descriptions = VariableDescriptionMap(self)
        super(EquationBase, self).__init__()
        method_list = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]
        for method in method_list:
            method_call = getattr(self, method)
            if hasattr(method_call, '_equation'):
                self.equations.append(method_call)


    def add_parameter(self, tag, init_val):
        """

        Parameters
        ----------
        tag
        init_val

        Returns
        -------

        """
        self.add_variable(tag, init_val, VariableType.PARAMETER)

    def add_constant(self, tag, value):
        """

        Parameters
        ----------
        tag
        value

        Returns
        -------

        """
        self.add_variable(tag, value, VariableType.CONSTANT)

    def add_state(self, tag, init_val):
        """

        Parameters
        ----------
        tag
        init_val

        Returns
        -------

        """
        if not isinstance(init_val, float) and not isinstance(init_val, int):
            raise ValueError("State must be float or integer")
        self.add_variable(tag, init_val, VariableType.STATE)
        self.add_variable(tag + '_dot', 0, VariableType.DERIVATIVE)

    def add_variable(self, tag, init_val, var_type):
        """

        Parameters
        ----------
        tag
        init_val
        var_type

        Returns
        -------

        """
        self.variables_descriptions. \
            register_variable_description(VariableDescription(tag=tag, id=str(uuid.uuid1()), initial_value=init_val,
                                                              type=var_type))
