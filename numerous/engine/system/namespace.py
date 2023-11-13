import logging
import uuid
import deprecation

from numerous.multiphysics import EquationBase
from numerous.utils.dict_wrapper import _DictWrapper
from numerous.engine.variables import Variable, VariableDescription, _VariableFactory, SetOfVariables


class VariableNamespaceBase:
    """
    Represents a set of variables.

    """

    def __init__(self, item, tag, is_connector=False, _id=uuid.uuid1()):
        self.items = [item.id]
        self.is_connector = is_connector
        self.item = item
        self.set_variables = []
        self.id = str(_id)
        # self.variable_scope = []
        ## -1 outgoing
        ## 0 no mapping
        ## Currently only used in SetNamespace
        self.mappings = []
        self.tag = tag
        self.path = [tag]
        self.outgoing_mappings = 0
        self.associated_equations = {}
        self.variables = _DictWrapper(self.__dict__, Variable)
        self.variable_scope = [self.variables]
        self.registered = False
        self.part_of_set = False
        ## if it is part of set it will be ignored during assembly.
        # print('creating ns with tag: ',tag)

    def get_path_dot(self):
        return ".".join(self.path)

    def __getitem__(self, y):
        return self.variables[y]

    def __setattr__(self, name, value):
        if isinstance(value, Variable):

            self.outgoing_mappings += 1
            self[name].add_mapping(value)
        else:
            object.__setattr__(self, name, value)

    @deprecation.deprecated(deprecated_in="0.3", removed_in="0.4",
                            details='use add_constant, add_parameter, or add_state')
    def create_variable(self, name, value=0):
        """
        Creates a variable in the namespaces with given name.

        Parameters
        ----------
        name: string
            Name of a 'Variable'

        """
        self.create_variable_from_desc(VariableDescription(tag=name, initial_value=value))

    def create_variable_from_desc(self, variable_description):
        """
        Creates and register a variable from given description.

        Parameters
        ----------
        variable_description: 'VariableDescription'
            variable_description
        on_assign_overload : 'OverloadAction'
            action on assign overload
               """

        name = variable_description.tag
        if variable_description.update:

            self[name].model.write_variables(variable_description.initial_value, self[name].llvm_idx)
        else:

            variable = _VariableFactory._create_from_variable_desc(self, self.item, variable_description)
            variable_description._variable = variable
            self.register_variable(variable)

    def get_flat_variables(self):
        return self.variable_scope

    def get_variable(self, var_tag):
        """
        Get a variable with given description.

        Parameters
        ----------
        var_description: 'VariableDescription'
            variable_description

         Returns
        -------
        variable : 'Variable'
                returns a variable matching description or None

        """
        if var_tag in self.variables.internal_dict.keys():
            return self.variables.internal_dict[var_tag]
        else:
            return None

    def register_variable(self, variable, tag_count=""):
        """
        Registering existing Variable in the namespace.

        Parameters
        ----------
        variable: 'Variable'
            Variable to be registered.

        """
        if (variable.tag + tag_count) not in self.variables.keys():
            self.variables[variable.tag + tag_count] = variable

            variable.path.extend_path(variable.id, self.id, self.tag)
            variable.path.extend_path(self.id, self.item.id, self.item.tag)

        else:
            logging.warning("Variable {0} is already in namespace {1} of item {2}".format(variable.tag,
                                                                                          self.tag, self.item.tag))
            # we overwrite constant < parameters < state
            if self.variables[variable.tag].value < variable.value:
                self.variables[variable.tag] = variable
                variable.extend_path(self.tag)
                variable.extend_path(self.item.tag)

    def update_set_var(self):
        pass

    def clear_equations(self):
        self.associated_equations = {}

    def add_equations(self, list_of_equations: list[EquationBase], update_bindings: bool = True,
                      create_variables: bool = True, set_equation: bool = False):
        """
        Adding a list of equations to namespace. Each equation in the list is parsed and all
         required variables are created and registered in the namespace.

         Parameters
        ----------
        list_of_equations : list of 'EquationBase'
            List of the equations to be added to the namespace.
        update_bindings : bool, optional
             if True creates and register a binding variables in all bindings associated with item
             that namespace is created in.
        create_variables : bool, optional
            Create variables defined in the 'EquationBase' inside namespace, by default True
        set_equation : bool, optional
            Only used when registering set of items, by default False
        """
        for eq in list_of_equations:
            self.add_equation(eq, update_bindings, create_variables, set_equation)

    def add_equation(self, equation: EquationBase, update_bindings: bool = True, create_variables: bool = True,
                     set_equation: bool = False):
        """
        Add an equation to a namespace.
        Parameters
          ----------
        equation : 'EquationBase'
            The equation to add to the namespace.
        update_bindings : bool, optional
             if True creates and register a binding variables in all bindings associated with item
             that namespace is created in.
        create_variables : bool, optional
            Create variables defined in the 'EquationBase' inside namespace, by default True
        set_equation : bool, optional
            Only used when registering set of items, by default False
        """
        if update_bindings and self.is_connector:
            self.item.update_bindings([equation], self.tag)

        if create_variables:
            any(self.create_variable_from_desc(variable_description)
                for variable_description in equation.variables_descriptions)
        equation.set_equation = set_equation
        self.associated_equations.update({equation.tag: equation})

    def set_values(self, **kwargs):
        """
            Set values of variables in the namespace by passing keyword arguments corresponding to variable names.
        """
        for k, v in kwargs.items():
            self.variables[k].value = v

class VariableNamespace(VariableNamespaceBase):
    pass


class SetNamespace(VariableNamespace):
    def __init__(self, item, tag, item_indcs):
        super().__init__(item, tag)
        self.tag = tag
        self.items_id = item_indcs
        self.items = []
        self.len_items = len(self.items)
        self.set_variables = {}

    def add_item(self, item, ns):
        self.items.append(item)
        item_ix = self.items_id.index(ns.item.id)
        for variable in ns.variables:
            if variable.tag not in self.set_variables.keys():
                self.set_variables[variable.tag] = SetOfVariables(variable.tag, self.item.tag, ns.tag)
            self.set_variables[variable.tag].add_variable(variable)
            variable.set_var_ix = item_ix

    def get_flat_variables(self):
        flat_variables = []
        for set_of_variables in self.set_variables.values():
            flat_variables.append(set_of_variables)
        return flat_variables


class _BindingVariable(Variable):
    def __init__(self, variable):
        super().__init__(variable.detailed_description)
        self.__dict__ = variable.__dict__


class _ShadowVariableNamespace(VariableNamespaceBase):
    def __init__(self, item, tag, binding, is_connector=False, _id=uuid.uuid1()):
        super().__init__(item, tag, is_connector, _id)
        self.binding = binding

    def register_variable(self, variable):
        if variable.tag not in self.variables:
            self.variables[variable.tag] = _BindingVariable(variable)
            self.variables[variable.tag].path.extend_path(variable.id, self.id, self.tag)
