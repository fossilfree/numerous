from numerous.engine.system.connector import Connector
from numerous.engine.system.item import Item
from numerous.engine.system.namespace import _BindingVariable


class ConnectorItem(Item, Connector):
    """
    Item that can be used as a connector.
    """

    def __init__(self, tag):
        super(ConnectorItem, self).__init__(tag=tag)

    def create_namespace(self, namespace_name):
        """
        Creating a namespace in item and all bindings.

        Parameters
        ----------
        namespace_name : string
            Name of a `VariableNamespace`

        Returns
        -------
        new_namespace : `VariableNamespace`
            Empty namespace with given name
        """
        new_namespace = super().create_namespace(namespace_name)
        super()._create_shadow_namespace(namespace_name)
        return new_namespace

    def __bind_mappings(self, binding, binded_item):
        for ns in self.registered_namespaces:
            for f_var in ns.variables:
                for i,b_fvar in enumerate(f_var.mapping):
                    if isinstance(b_fvar, _BindingVariable):
                        if b_fvar.namespace.binding.name == binding.name:
                            bv = binded_item.registered_namespaces[ns.tag].get_variable(
                                b_fvar.detailed_description)
                            f_var.mapping[i] = bv

    def bind(self, **kwargs):
        """
        Method to bind item to the bindings in current item. Biding items creating all mappings that
        a linked to Binding.

        Parameters
        ----------
        **kwargs : `Item`
            items to bind in form, binding_name = Item

        """
        for key, value in kwargs.items():
            if key in self.bindings.keys():

                self.bindings[key].add_binding(value)
                self.__bind_mappings(self.bindings[key], value)
            else:
                ValueError("Binding {} is not exist in item {}".format(key, self.tag))


class ConnectorTwoWay(ConnectorItem):
    """
        Special case of  a connector item with 2 predefined bindings.

    """

    def __init__(self, tag, side1_name='side1', side2_name='side2'):
        super(ConnectorTwoWay, self).__init__(tag=tag)
        self.binding_names = [side1_name, side2_name]
        for bn in self.binding_names:
            self.create_binding(bn)
