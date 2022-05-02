# from numerous.engine.model.events import NumerousEvent
from numerous.engine.numerous_event import NumerousEvent
from numerous.engine.system.connector import Connector
from numerous.utils.dict_wrapper import _DictWrapper
from numerous.engine.system.namespace import VariableNamespace, VariableNamespaceBase
from numerous.engine.system.node import Node
from numerous.utils.logger_levels import LoggerLevel

DEFAULT_NAMESPACE = 'default'


class Item(Node):
    """
           Base class for items hierarchy. Contains fields and methods required to process
           item objects. An Item in a Numerous engine are represent any
           object needed for simulation model.


          Attributes
          ----------
          registered_namespaces :  dictionary of `VariableNamespaceBase`
              Dictionary of namespaces registered in the current item. All registered namespaces are although added to
              to the __dict__ of an object and can be referenced as a class attribute.

       """

    def __init__(self, tag=None, logger_level=None):
        self.registered_namespaces = _DictWrapper(self.__dict__, VariableNamespaceBase)
        self.events = []
        self.level = 1
        self.parent_item = None
        self.registered = False
        self.parent_set = None
        self.part_of_set = False

        self.logger_level = logger_level
        if tag:
            self.tag = tag

        self.path = [self.tag]
        super(Item, self).__init__(self.tag)

    def get_default_namespace(self):
        """
        Returns a new namespace with name default for the current item. This namespace is not
        registered.

        Returns
        -------
        namespace : `VariableNamespaceBase`
            a default namespace.

        Examples
        --------
        >>> item = Item('example')
        >>> dn = item.get_default_namespace()
        >>> print(dn.item is None )
        True
        >>> item.register_namespace(dn)
        >>> print(dn.item is None )
        False
        """
        # return VariableNamespace(self, DEFAULT_NAMESPACE, is_connector=isinstance(self, Connector))
        return self.create_namespace(DEFAULT_NAMESPACE)

    def create_namespace(self, tag):
        """
        Creating a namespace.

        Parameters
        ----------
        tag : string
            Name of a class:`numerous.engine.VariableNamespace`

        Returns
        -------
        new_namespace : class:`numerous.engine.VariableNamespace`
            Empty namespace with given name

        """
        new_namespace = VariableNamespace(self, tag, is_connector=isinstance(self, Connector))
        self.register_namespace(new_namespace)
        return new_namespace

    def register_namespace(self, namespace):
        """
        Registering an already existed namespace for item.

        Parameters
        ----------
            namespace : `VariableNamespace`
                namespace to be registered.

        Raises
        ------
        ValueError
            If namespace is already registered for this item.
        """
        if not namespace.registered:
            if namespace.tag in self.registered_namespaces.keys():
                raise ValueError('Namespace with name {0} is already registered in item {1}'
                                 .format(namespace.tag, self.tag))
            else:
                self.registered_namespaces.update({namespace.tag: namespace})
            namespace.registered = True
        else:
            raise ValueError(f'Cannot register namespace {namespace.tag} more than once!')

    def get_variables(self):
        """
        Get variables from registered namespaces.

        Returns
        -------
        variables : list of tuples
            all variables with corresponding registered namespace. In for of tuple (variable,namespace).

        """
        variables_result = []
        for vn in self.registered_namespaces.values():
            for variable in vn.variables:
                variables_result.append((variable, vn))
        return variables_result

    def add_event(self, key, condition, action, compiled_functions=None, terminal=True, direction=1, compiled=False):
        condition = condition
        action = action
        event = NumerousEvent(key, condition, action, compiled, terminal, direction,
                              compiled_functions=compiled_functions)
        self.events.append(event)

    def _increase_level(self):
        self.level = self.level + 1

    def get_item(self, item_path):
        """
        Get an item using item path.

        Parameters
        ----------
        item_path: 'ItemPath'
            hierarchical path to an item inside the subsystem

        Returns
        -------
        Item : 'Item'
                returns an item found at given path or None

        """
        current_item = item_path.get_top_item()
        next_item_path = item_path.get_next_item_path()
        if self.tag == current_item and next_item_path:
            return None
        elif self.tag == current_item and next_item_path is None:
            return self
        else:
            return None

    def __getattr__(self, name):
        if DEFAULT_NAMESPACE not in self.registered_namespaces.keys():
            self.create_namespace(DEFAULT_NAMESPACE)

        namespace = getattr(self, DEFAULT_NAMESPACE)

        if name == DEFAULT_NAMESPACE:
            return namespace

        return getattr(namespace, name)

    def set_logger_level(self, logger_level=None):
        self.logger_level = logger_level
