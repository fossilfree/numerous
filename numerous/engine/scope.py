from .variables import Variable, VariableType, MappedValue
import numpy as np

class ScopeVariable(MappedValue):
    """
               Variable inside the scope.
    """

    def __init__(self, base_variable):
        super().__init__(base_variable.id)
        self.updated = False
        self.mapping_id = base_variable.mapping.id if base_variable.mapping else None

        if base_variable.sum_mapping is not None:
            self.sum_mapping_ids = [x.id for x in base_variable.sum_mapping]
        else:
            self.sum_mapping_ids = None

        self.value = base_variable.get_value()
        self.type = base_variable.type
        self.tag = base_variable.tag
        self.state_ix = None
        self.associated_state_scope = []
        self.bound_equation_methods = None
        self.parent_scope_id = None
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

    def update_mappings(self, model):
        self.flat_var[model.mapping_from] = self.flat_var[model.mapping_to]

    def update_states(self, state_values):
        np.put(self.flat_var, self.state_idx, state_values)

    def update_states_idx(self, state_value, idx):
        np.put(self.flat_var, idx, state_value)

    # return all derivatives
    def get_derivatives(self):
        return self.flat_var[self.deriv_idx]

    # return derivate of state at index idx
    def get_derivatives_idx(self, idx):
        return self.flat_var[idx]
