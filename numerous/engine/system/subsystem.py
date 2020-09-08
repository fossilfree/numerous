import copy
from enum import Enum

from numerous import EquationBase
from numerous.utils.dict_wrapper import _DictWrapper
from numerous.engine.system.item import Item
import networkx as nx
from numerous.engine.system.connector_item import ConnectorItem
from numerous.engine.system.namespace import SetNamespace


class ItemsStructure(Enum):
    LIST = 0
    GRID = 1
    SET = 2


class Subsystem(ConnectorItem):
    """
    Hierarchical representation of a group of items. Subsystems contains  a set of registered items, some
    of such items can be subsystems itself. This allow us to create Subsystems of any complexity.

    """

    def __init__(self, tag):
        self.ports = _DictWrapper({}, Item)
        self.registered_items = {}
        super().__init__(tag)

    def add_port(self, port_tag, item):
        """
        Creates a port for a subsystem. Ports can be used as an elements for binding.

        Parameters
        ----------
        port_tag: string
            Name of the port.
        item: :class:`numerous.engine.system.Item`
            item that is be used as a port.
        """

        if item.id not in self.registered_items.keys():
            raise ValueError(
                "Item {0} cannot be registered as port for subsystem {1}. Item is not registered in subsystem"
                    .format(item.tag, self.tag))
        self.ports[port_tag] = item

    def get_item(self, item_path):
        """
        Get an item using item path.

        Parameters
        ----------
        item_path: :class:`numerous.engine.system.ItemPath`
            hierarchical path to an item inside the subsystem

        Returns
        -------
        Item : :class:`numerous.engine.system.Item`
                returns an item found at given path or None

        """
        current_item = item_path.get_top_item()
        next_item_path = item_path.get_next_item_path()
        if self.tag == current_item and next_item_path:
            for item in self.registered_items.values():
                if item.tag == next_item_path.get_top_item():
                    return item.get_item(next_item_path)
            else:
                return None
        elif self.tag == current_item and next_item_path is None:
            return self
        else:
            return None

    def register_items(self, items, tag="set", structure=ItemsStructure.LIST):
        """

        Parameters
        ----------
        items : list of :class:`numerous.engine.system.Item`
            List of items to register in the subsystem.
        """
        if structure == ItemsStructure.LIST:
            any(self.register_item(item) for item in items)
        elif structure == ItemsStructure.SET:
            self.register_item(ItemSet(items, tag))
            any(self.register_item(item) for item in items)

    def increase_level(self):
        super().increase_level()
        for item in self.registered_items.values():
            item.increase_level()

    def get_graph_visualisation(self, DG=None, parent=None):
        if DG is None:
            DG = nx.DiGraph()
        DG.add_node(self.tag)
        if parent:
            DG.add_edge(parent, self.tag)
        for item in self.registered_items.values():
            if isinstance(item, Subsystem):
                item.get_graph_visualisation(DG, self.tag)
            else:
                DG.add_edge(self.tag, item.tag)
        return DG

    def update_variables_path(self, item):
        item.path = self.path + [item.tag]
        for ns in item.registered_namespaces.values():

            ns.path = item.path + [ns.tag]
            for var in ns.variables.values():
                var.path.extend_path(item.id, self.id, self.tag, registering=True)
                var.path_ = ns.path + [var.tag]
                var.paths.append(var.path_)
        if isinstance(item, Subsystem):
            for item_ in item.registered_items.values():
                item.update_variables_path(item_)

    def update_variables_path_(self):
        #item.path = self.path + [item.tag]
        for ns in self.registered_namespaces.values():

            ns.path = self.path + [ns.tag]
            for var in ns.variables.values():
                var.path.extend_path(self.id, self.id, self.tag, registering=True)
                var.path_ = ns.path + [var.tag]
                var.paths.append(var.path_)
        if isinstance(self, Subsystem):
            for item_ in self.registered_items.values():
                if isinstance(item_, Subsystem):
                    item_.update_variables_path_()

    def register_item(self, item):
        """

        Parameters
        ----------
        item : :class:`numerous.engine.system.Item`
            Item to register in the subsystem.
        """
        #if not item.registered:
        if item.tag in [x.tag for x in self.registered_items.values()]:
            raise ValueError('Item with tag {} is already registered in system {}'.format(item.tag, self.tag))
        item._increase_level()
        self.update_variables_path(item)

        self.registered_items.update({item.id: item})
        #else:
        #    raise ValueError(f'Cannot register item {item.tag} more than once!')


class ItemSet(Item, EquationBase):

    def __init__(self, set_structure, tag):
        super().__init__(tag)

        set_structure_flat = set_structure#.flatten()

        self.item_ids = []
        for item in set_structure_flat:
            self.item_ids.append(item.id)

        tag_count = 0
        ##TODO Check that all items are of the same type
        for item in set_structure_flat:
            for ns in item.registered_namespaces:
                print('tag!: ',tag)
                tag_ = ns.tag
                if not (tag_ in self.registered_namespaces.keys()):
                    print(tag_)
                    sns = SetNamespace(self, tag_, self.item_ids)

                    sns.add_equations(list(ns.associated_equations.values()), False)
                    self.register_namespace(sns)
                self.registered_namespaces[tag_].add_item_to_set_namespace(ns, tag_count)
                if not ns.part_of_set:
                    ns.part_of_set = sns
                else:
                    ValueError(f'namespace {ns} already in set {ns.part_of_set}')
                tag_count += 1

                #item.registered_namespaces.pop(ns)
        for ns in self.registered_namespaces.values():
            ns.items = self.item_ids
