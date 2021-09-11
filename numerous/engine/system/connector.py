from numerous.engine.system.binding import Binding
from numerous.utils.dict_wrapper import _DictWrapper
from numerous.engine.system.namespace import _ShadowVariableNamespace
from numerous.engine.system.node import Node



class Connector(Node):
    """
     Base class for representing connectors. Object that inherited connector can be used as
     a connection between items.

     Attributes
     ----------
          bindings :  dictionary of :class:`Binding`
               List of binding that connector have.
    """

    def __init__(self, tag, **kw):
        self.bindings = _DictWrapper(self.__dict__, Binding)
        super(Connector, self).__init__(tag)

    def create_binding(self, binding_name):
        """
        Creating a new binding inside the connector

        Parameters
        ----------
            binding_name : string
                name of the new binding

        Raises
        ------
        ValueError
            If `binding_name` is already registered in this connector.
        """
        if binding_name in self.bindings.keys():
            raise ValueError('Binding with name {0} is already registered in connector {1}'
                             .format(binding_name, self.tag))
        else:
            self.bindings[binding_name] = Connector.create_new_binding(binding_name)

    def _create_shadow_namespace(self, param):
        self.__update_bindings_namespace(param)

    def update_bindings(self, list_of_eq, binding_name):
        """
        Updating an existing binding with the equations that are expected to be in the binded items.

        Parameters
        ----------
        list_of_eq : list of :class:`numerous.multiphysics.Equation`
            List of a `Equation` that are expected to be in binded items.

        binding_name: string
            Name of a binding to be updated.
        """
        for binding in self.bindings:
            binding.__dict__[binding_name].add_equations(list_of_eq)

    def __update_bindings_namespace(self, param):
        for binding in self.bindings:
            binding.update_namespace(_ShadowVariableNamespace(self, param, binding))

    def get_binded_items(self):
        """
              Get items that are binded to the connector.

              Returns
              -------
              items : list
                  all items that are binded to the connector as one list.

        """
        return [
            y.binded_item for y in self.bindings
            if y.binded_item is not None
        ]

    @staticmethod
    def create_new_binding(binding_name):
        """
        Creates a new :class:`Binding` without registering it inside the connector.

        Parameters
        ----------
            binding_name: string
                Name of a binding to be created.

        Returns
        -------
            binding : Binding
                  new binding with given name.

        """
        return Binding(binding_name)

    def __setattr__(self, key, value):
        if key in self.__dict__:
            if isinstance(self.__dict__[key], Binding):
                if isinstance(value, Node):
                    self.__dict__[key].add_binding(value)
        object.__setattr__(self, key, value)
