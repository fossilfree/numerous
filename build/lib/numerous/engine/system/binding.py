from numerous.utils.dict_wrapper import _DictWrapper
from .namespace import VariableNamespaceBase


class _NamespaceManagerList:
    def __init__(self, ns):
        self.__dict__[ns.tag] = ns

    def add_namespace(self, ns):
        self.__dict__[ns.tag] = ns


class Binding:
    """
        Represents expected binding of an `Item`. When binding is point to an actual item it checks for
        mapped variables and namespaces.

        Attributes
        ----------
             binding_name :  string
    """

    def __init__(self, binding_name):
        self.name = binding_name
        self.ns = _DictWrapper(self.__dict__, VariableNamespaceBase)
        self.binded_item = None

    def update_namespace(self, ns):
        """
            Updating an existing or creating a new namespace.

            Parameters
            ----------
            ns : :class:`Item`
                Namespace.
        """
        self.ns[ns.tag] = ns

    def __create_binding_varaible_bindings(self, namespace):
        if namespace.tag in self.binded_item.registered_namespaces.keys():
            for bv in namespace.variables:
                for mapping in  bv.mapping:
                    self.binded_item.registered_namespaces[namespace.tag].variables[bv.tag].add_mapping(mapping)

    def __merge_namespaces(self):
        for namespace in self.ns:
            self.__create_binding_varaible_bindings(namespace)
            self.ns[namespace.tag] = self.binded_item.registered_namespaces[namespace.tag]

    def add_binding(self, item):
        """
            Add an 'Item' to binded items.

            Parameters
            ----------
            item : :class:`Item`
                Item to be used as a binding.
        """
        if self.binded_item:
            if self.binded_item.id == item.id:
                raise ValueError("item {0} is already binded to binding {1}".format(item.tag, self.name))
            else:
                raise ValueError("Binding multiple items to the same binding is not supported")
        else:
            self.binded_item = item
            self.__merge_namespaces()

    def is_bindend(self):
        """
            Checks if item is binded to this binding.

            Returns
            -------
            is_bindend : bool
              """
        return not (self.binded_item is None)