import logging
import uuid
import inspect
from numerous.utils.dict_wrapper import _DictWrapper
from numerous.engine.variables import Variable, VariableDescription, _VariableFactory, OverloadAction


class VariableNamespaceBase:
    """
    Represents a set of variables.

    """

    def __init__(self, item, tag, is_connector=False, _id=uuid.uuid1()):
        self.is_connector = is_connector
        self.item = item
        self.id = str(_id)
        self.tag = tag
        self.outgoing_mappings = 0
        self.associated_equations = {}
        self.variables = _DictWrapper(self.__dict__, Variable)

    def __getitem__(self, y):
        return self.variables[y]

    def __setattr__(self, name, value):
        if isinstance(value, Variable):

            self.outgoing_mappings += 1
            # curframe = inspect.currentframe()
            # calframe = inspect.getouterframes(curframe, 2)
            # print(name,' mapped to: ', value, ' by ',calframe[1][3])
            self[name].add_mapping(value)
        else:
            object.__setattr__(self, name, value)

    def create_variable(self, name):
        """
        Creates a variable in the namespaces with given name.

        Parameters
        ----------
        name: string
            Name of a 'Variable'

        """
        self.create_variable_from_desc(VariableDescription(tag=name))

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
        variable = _VariableFactory._create_from_variable_desc(self, self.item, variable_description)
        self.register_variable(variable)

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

    def register_variable(self, variable):
        """
        Registering existing Variable in the namespace.

        Parameters
        ----------
        variable: 'Variable'
            Variable to be registered.

        """
        if variable.tag not in self.variables:
            self.variables[variable.tag] = variable

            variable.path.extend_path(variable.id, self.id, self.tag)
            variable.path.extend_path(self.id, self.item.id, self.item.tag)
        else:
            logging.warning("Variable {0} is already in namespace {1} of item {2}".format(variable.tag,
                                                                                          self.tag, self.item.tag))
            # we overwrite constant < parameters < state
            if self.variables[variable.tag].type < variable.type:
                self.variables[variable.tag] = variable
                variable.extend_path(self.tag)
                variable.extend_path(self.item.tag)

    def add_equations(self, list_of_equations, update_bindings=True):
        """
        Adding a list of equations to namespace. Each equation in the list is parsed and all
         required variables are created and registered in the namespace.

        Parameters
        ----------
        list_of_equations: list of 'Equation'
            list of equations to be added

        update_bindings: bool
            if True creates and register a binding variables in all bindings associated with item
             that namespace is created in.


        """
        if update_bindings and self.is_connector:
            self.item.update_bindings(list_of_equations, self.tag)
        for eq in list_of_equations:
            any(self.create_variable_from_desc(variable_description)
                for variable_description in eq.variables_descriptions)

            self.associated_equations.update({eq.tag: eq})


class VariableNamespace(VariableNamespaceBase):
    pass


class SetNamespace(VariableNamespace):
    def __init__(self, item, tag):
        super().__init__(item, tag)
        self.variable_scope = []
        ## -1 outgoing
        ## 0 no mapping
        ##
        self.mappings = []

    def add_item_to_set_namespace(self, item):
        mapping = []
        variables = []
        for variable in item.variables:
            self.register_variable(variable)
            if variable.mapping:
                mapping.append(-1)
            else:
                mapping.append(0)
            variables.append(variable)
        self.variable_scope.append(variables)
        self.mappings.append(mapping)



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
