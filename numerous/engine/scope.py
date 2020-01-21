from .variables import Variable, VariableType
from numerous.utils.dict_wrapper import _DictWrapper
from collections import ChainMap


class ScopeVariable:
    """
               Variable inside the scope.
    """

    def __init__(self, base_variable):
        self.updated = False
        # self.scope = None
        self.mapping_ids = [var.id for var in base_variable.mapping]
        self.mapping = []
        self.path = base_variable.path
        self.value = base_variable.value
        self.type = base_variable.type
        self.tag = base_variable.tag
        self.id = base_variable.id
        self.state_ix = None
        self.associated_state_scope = []

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
            return self.variables[item].value
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

    def __init__(self, scope_dict, states):
        self.scope_dict = scope_dict
        self.states = states
        self.result = {}

    def update_model_from_scope(self, model):
        for scope in self.scope_dict.values():
            for scope_var in scope.variables.values():
                if model.variables[scope_var.id].value != scope_var.value:
                    model.variables[scope_var.id].value = scope_var.value

    def get_scope_vars(self, state):
        if state.id in self.result:
            return self.result[state.id]
        else:
            self.result[state.id] = []
            for scope in [self.scope_dict[your_key] for your_key in state.associated_state_scope]:
                var = scope.variables[state.tag]
                if var.id == state.id:
                    self.result[state.id].append((var, scope))
        return self.result[state.id]

    def update_states(self, state_values):
        for i, state in enumerate(self.states.values()):
            scope_vars = self.get_scope_vars(state)
            value = state_values[i]
            for var, scope in scope_vars:
                scope.variables[var.tag].value = value
                var.value = value

    def get_derivatives(self):
        def scope_derivatives_dict(scope):
            return {var.id: var \
                    for var in scope.variables.values() \
                    if var.type.value == VariableType.DERIVATIVE.value}

        # list of dictionaries
        resList = list(map(scope_derivatives_dict, self.scope_dict.values()))
        # one dictionary, list must be reversed because:
        # in the old code the key,values  were updated from left to right
        # chainMap only keeps the first encoutnered key,value
        return ChainMap(*(reversed(resList))).values()

    def update_mappings_and_time(self, timestamp=None):
        # input scope_dict
        # output a new updated dict
        def new_time_map(scope, t):
            return {key: new_scope_value_time_and_mapping(val, t) \
                    for (key, val) in scope.items()}

        # input a scope_dict value
        # output a new value with updated time/ variables
        def new_scope_value_time_and_mapping(scope_v, t):
            # clone the old value in order to keep things "immutable"
            val = scope_v
            val.set_time(t)
            val.variables.update(new_variable_mapping(val))
            return val

        # input a scope_dict value
        # returns a dictionary corespnding to the values variables
        def new_variable_mapping(scope_v):
            return {var.tag: var \
                    for var in scope_v.variables.values()}

        # updates the dictionary
        self.scope_dict.update( \
            new_time_map(self.scope_dict, timestamp))
