import uuid
from collections import deque
import numpy as np
import numbers

import numerous.declarative.equations
from numerous.engine.variables import VariableBase, VariableDescription, VariableType
from numerous.multiphysics.equation_decorators import InlineEquation


class VariableDescriptionMap(VariableBase):
    def __init__(self, equation):
        self.variables_descriptions = super().__dict__
        self.equation = equation
        self.variables_descriptions_deque = deque()

    def variable_exists(self, tag):
        return tag in self.variables_descriptions

    def register_variable_description(self, variable_description):
        if variable_description.tag not in self.variables_descriptions:
            self.variables_descriptions[variable_description.tag] = variable_description
        else:
            self.variables_descriptions[variable_description.tag].update = True
            return
            raise ValueError('Variable description with tag {} is already exist in equation {}'.format(
                variable_description.tag, numerous.dec.equations.equation.tag))

    # refactored in a more functional way
    def __iter__(self):
        # used filter and extend
        self.variables_descriptions_deque.extend(list(filter((lambda v: isinstance(v, VariableDescription)),
                                                             self.variables_descriptions.values())))
        return self

    def __next__(self):
        if self.variables_descriptions_deque:
            return self.variables_descriptions_deque.pop()
        else:
            raise StopIteration


class Mapping:
    def __init__(self, from_, to_=None, dir='<-'):
        self.from_ = from_
        self.to_ = to_ if to_ is not None else from_
        self.dir = dir


class EquationBase:
    """
    Equations governing the behavior of the objects in the model and their interactions
     are defined in classes extending the :class:`numerous.multiphysics.Equation`.

    """

    def __init__(self, tag=None):
        if tag:
            self.tag = tag
        self.equations = []
        self.new_variable_idx = 0
        self.variables_descriptions = VariableDescriptionMap(self)
        super(EquationBase, self).__init__()
        method_list = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__")]
        for method in method_list:
            method_call = getattr(self, method)
            if hasattr(method_call, '_equation'):
                self.equations.append(method_call)

    def add_parameter(self, tag, init_val=0, logger_level=None, alias=None, integrate=None):
        """

        Parameters
        ----------
        tag
        init_val

        Returns
        -------

        """
        var_desc = self.add_variable(tag, init_val, VariableType.PARAMETER, logger_level, alias)

        if integrate is not None:
            self.add_state(integrate['tag'], 0, logger_level=logger_level)

            integrate_source = f"""
            def integrate_{tag}(self, scope):
                scope.{integrate['tag']}_dot = scope.{tag} * {integrate['scale']}
            """
            ie = InlineEquation()
            integrate_source_ = ie('integrate_' + tag, integrate_source, {})

            setattr(self, 'integrate_' + tag, integrate_source_)
            self.equations.append(integrate_source_)

        return var_desc

    def add_parameters(self, parameters: dict or list):
        if isinstance(parameters, dict):
            for p, v in parameters.items():
                if isinstance(v, numbers.Number):
                    self.add_parameter(p, init_val=v)
                else:
                    self.add_parameter(p, **v)
        if isinstance(parameters, list):
            for p in parameters:
                self.add_parameter(p)

    def add_constants(self, constants: dict):
        for p, v in constants.items():
            if isinstance(v, numbers.Number):
                self.add_constant(p, value=v)
            else:
                self.add_constant(p, **v)

    def add_constant(self, tag, value, logger_level=None, alias=None):
        """

        Parameters
        ----------
        tag
        value

        Returns
        -------

        """
        self.add_variable(tag, value, VariableType.CONSTANT, logger_level, alias)

    def add_derivative(self, tag, logger_level=None, alias=None):
        self.add_variable(tag + '_dot', 0, VariableType.DERIVATIVE, logger_level,
                          alias + "_dot" if alias is not None else None)

    def add_state(self, tag, init_val=0, logger_level=None, alias=None, create_derivative=True):
        """

        Parameters
        ----------
        tag
        init_val
        create_derivative

        Returns
        -------

        """
        if not isinstance(init_val, float) and not isinstance(init_val, int) and not isinstance(init_val, np.int64):
            raise ValueError("State must be float or integer")
        self.add_variable(tag, init_val, VariableType.STATE, logger_level, alias)
        if create_derivative:
            self.add_variable(tag + '_dot', 0, VariableType.DERIVATIVE, logger_level,
                              alias + "_dot" if alias is not None else None)

    def add_variable(self, tag, init_val, var_type, logger_level, alias):
        """

        Parameters
        ----------
        tag
        init_val
        var_type
        external_mappng

        Returns
        -------

        """
        var_description = VariableDescription(tag=tag, id=str(uuid.uuid1()), initial_value=init_val,
                                                              type=var_type, logger_level=logger_level, alias=alias,
                                                              variable_idx=self.new_variable_idx)
        self.variables_descriptions. \
            register_variable_description(var_description)
        self.new_variable_idx += 1

        return var_description

    def map_create_parameters(self, item, mappings):
        for m in mappings:
            if not self.variables_descriptions.variable_exists(m.side2):
                self.add_parameter(m.side2)
