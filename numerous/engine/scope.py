from .variables import Variable, VariableType
from numerous.utils.dict_wrapper import _DictWrapper


class ScopeVariable(Variable):
    """
               Variable inside the scope.
    """

    def __init__(self, base_variable, scope):
        super().__init__(base_variable.detailed_description, base_variable)
        self.base_variable = base_variable
        self.updated = False
        self.scope = scope
        self.mapping_ids = [var.id for var in base_variable.mapping]
        self.mapping = []


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

    def __init__(self):
        self.variables = _DictWrapper(self.__dict__, object)
        self.variables_dict = {}
        self.globals = GlobalVariables(0)

    def set_time(self, time):
        self.globals.time = time

    def add_variable(self, variable):
        """
            Function to add variables to the scope.

            Parameters
            ----------
            variable : Variable
                Original variable associated with namespace.

            """
        scope_var = ScopeVariable(variable, self)
        self.variables.update({variable.tag: scope_var.value})
        self.variables_dict.update({variable.tag: scope_var})

    def __setattr__(self, key, value):
        if 'variables_dict' in self.__dict__:
            if key in self.variables_dict:
                self.variables_dict[key].update_value(value)
        object.__setattr__(self, key, value)

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

    def update_model_from_scope(self):
        for scope in self.scope_dict.values():
            for scope_var in scope.variables_dict.values():
                if scope_var.base_variable.value != scope_var.value:
                    scope_var.base_variable.value = scope_var.value

    def get_scope_vars(self, state):

        if state.id in self.result:
            return self.result[state.id]
        else:
            self.result[state.id] = []
            for scope in [self.scope_dict[your_key] for your_key in state.associated_scope]:
                var = scope.variables_dict[state.tag]
                if var.id == state.id:
                    self.result[state.id].append((var, scope))
        return self.result[state.id]

    def update_states(self, state_values):
        for i, state in enumerate(self.states.values()):
            scope_vars = self.get_scope_vars(state)
            value = state_values[i]
            for var, scope in scope_vars:
                scope.variables[var.tag] = value
                var.value = value

    # can be refactored the same way as update_mapping_and_time
    # TODO
    def get_derivatives(self):
        result = {}
        for scope in self.scope_dict.values():
            for var in scope.variables_dict.values():
                if var.type.value == VariableType.DERIVATIVE.value:
                    result.update({var.id:var})
                    # read: https://www.programiz.com/python-programming/methods/dictionary/update
                    # you can pass a dictionary and update it
                    # OLD CODE
                    # if var.id in result.keys():
                        # result[var.id].update(var)
                    # else:
                        # result.update({var.id: var})
        return result.values()

    # refactored the method to use dictionary comprehension
    def update_mappings_and_time(self, timestamp=None):
        # input scope_dict
        # output a new updated dict
        # has no correspondence in the iterative code
        # due to the fact that data is immutable
        def new_time_map(scope, t):
            return {key: new_scope_value_time_and_mapping(val,t)\
                for key, val in scope}

        # input a scope_dict value
        # output a new value with updated time/ variables
        # corresponds to 2. in the iterative code
        def new_scope_value_time_and_mapping(scope_v, t):
            # alternative way:
            #   scope_v.set_time(t);
            #   scope_v.variables.update(new_variable_mapping(scope_v))
            #   return scope_v
            # clone the old value in order to keep things "immutable"
            val = scope_v
            val.set_time(t)
            val.variables.update(new_variable_mapping(val))
            return val

        # input a scope_dict value
        # returns a dictionary corespnding to the values variables
        # corresponds to 1. in the iterative code
        def new_variable_mapping(scope_v):
            return {var.tag: var.value\
                for var in scope_v.variables_dict.values()}

        # updates the dictionary
        self.scope_dict.update(\
            new_time_map(self.scope_dict, timestamp))

    # OLD CODE
    # def update_mappings_and_time(self, timestamp=None):
        # "2. this loop updates scope_dict.value time+variable mapping"
        # for scope in self.scope_dict.values():
            # scope.set_time(timestamp)
            # "1. this loop updates scope_dict.value variable mapping"
            # for var in scope.variables_dict.values():
                # scope.variables[var.tag] = var.value