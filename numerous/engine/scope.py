from .variables import Variable, MappedValue
import numpy as np


class ScopeVariable(MappedValue):
    """
               Variable inside the scope.
    """

    def __init__(self, base_variable):
        super().__init__()
        self.updated = False
        self.mapping_ids = [var.id for var in base_variable.mapping]
        self.sum_mapping_ids = [var.id for var in base_variable.sum_mapping]
        self.value = base_variable.get_value()
        self.type = base_variable.type
        self.tag = base_variable.tag
        self.id = base_variable.id
        self.state_ix = None
        self.associated_state_scope = []
        self.position = None

    def update_ix(self, ix):
        self.state_ix = ix

    def update_value(self, value):
        self.value = value


class GlobalVariables:

    def __init__(self, time):
        self.time = time


class Scope:
    """
          Allows to collect a copy of variables relevant for the specific equation.

    Attributes
    ----------
          variables :  dict
               Variables relevant for the specific equation.

    """

    def __init__(self, scopeid):
        self.variables = {}
        self.variables_id = []
        self.id = scopeid
        self.globals = GlobalVariables(0)

    def set_time(self, time):
        self.globals.time = time

    def add_variable(self, scope_var):
        """
            Function to add variables to the scope.

            Parameters
            ----------
            variable : Variable
                Original variable associated with namespace.

            """
        # scope_var.add_scope(self)
        self.variables.update({scope_var.tag: scope_var})
        self.variables_id.append(scope_var.id)

    def __setattr__(self, key, value):
        if 'variables_dict' in self.__dict__:
            if key in self.variables_id:
                self.variables_id[key].update_value(value)
        if 'variables' in self.__dict__:
            if key in self.variables:
                self.variables[key].value = value
        object.__setattr__(self, key, value)

    def __getattribute__(self, item):
        if item == 'variables' or item == '__setstate__' or item == '__dict__':
            return object.__getattribute__(self, item)
        if item in self.variables:
            return self.variables[item].get_value()
        return object.__getattribute__(self, item)

    def apply_differential_equation_rules(self, is_true):
        """

         Set if equation allow to update scope variables

            Parameters
            ----------
            is_true : bool
                if we allow variable updates.
        """
        for var in self.variables_dict.values():
            var.allow_update = is_true


class TemporaryScopeWrapper:

    def __init__(self, flat_scope_var, state_idx,deriv_idx):
        self.flat_var = flat_scope_var
        self.state_idx = state_idx
        self.deriv_idx = deriv_idx
        self.result = {}
        self.name_idx = {}


    def update_states(self, state_values):
        np.put(self.flat_var, self.state_idx, state_values)

    def get_derivatives(self):
        return np.take(self.flat_var, self.deriv_idx)