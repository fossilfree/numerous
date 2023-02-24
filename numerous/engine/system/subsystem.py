from enum import Enum

from numba import types
from numba.typed import List

from numerous.engine.system.external_mappings import ExternalMappingUnpacked

from numerous.multiphysics import EquationBase
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

    def __init__(self, tag, external_mappings=None, data_loader=None, run_after_solve=None, post_step=None):
        self.ports = _DictWrapper({}, Item)
        self.registered_items = {}
        self.run_after_solve = run_after_solve if run_after_solve is not None else []
        self.post_step = post_step if post_step is not None else []
        self.external_mappings = ExternalMappingUnpacked(external_mappings, data_loader) if external_mappings else None
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

    def register_items(self, items, tag="set", structure=ItemsStructure.LIST, bind=False):
        """

        Parameters
        ----------
        items : list of :class:`numerous.engine.system.Item`
            List of items to register in the subsystem.
        """
        if structure == ItemsStructure.LIST:
            any(self.register_item(item) for item in items)
        elif structure == ItemsStructure.SET:
            itemset = ItemSet(items, tag)
            self.register_item(itemset)

        if bind:
            for i in items:
                setattr(self, i.tag, i)

    def _increase_level(self):
        super()._increase_level()
        for item in self.registered_items.values():
            item._increase_level()

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

    def update_variables_path(self, item, tail=[]):
        """
        Create variable path hierarchy.

        Parameters
        ----------
        item : :class:`numerous.engine.system.Item`
            Item to update variable path.
        tail:  list of :class:`numerous.engine.system.Item`
            In case of nested subsystems we store them to correctly update the path.

        """
        item.path = self.path + [item.tag]
        for ns in item.registered_namespaces.values():
            for var in ns.variables.values():
                var.path.extend_path(item.id, self.id, self.tag)
                if len(tail) == 1:
                    var.path.extend_path(self.id, tail[0].id, tail[0].tag)
                if len(tail) > 1:
                    t1 = tail[0::2][0]
                    t2 = tail[1::2][0]
                    var.path.extend_path(t2.id, t1.id, t1.tag)
        if isinstance(item, Subsystem):
            for item_ in item.registered_items.values():
                item.update_variables_path(item_, tail + [self])

    def register_item(self, item):
        """

        Parameters
        ----------
        item : :class:`numerous.engine.system.Item`
            Item to register in the subsystem.
        """
        if item.tag in [x.tag for x in self.registered_items.values()]:
            raise ValueError('Item with tag {} is already registered in system {}'.format(item.tag, self.tag))
        item._increase_level()
        self.update_variables_path(item)
        self.registered_items.update({item.id: item})

    def _find_variable(self, item, varname):
        variables = item.get_variables()
        for variable in variables:
            for path in variable[0].path.path.values():
                if path[0] == varname:
                    return True
        return False

    def find_variable(self, varname):
        if self._find_variable(self, varname):
            return True

        for item in self.registered_items.values():
            if isinstance(item, Subsystem):
                if item.find_variable(varname):
                    return True
            if self._find_variable(item, varname):
                return True
        return False

    def _get_external_mappings(self):
        external_mappings = []
        if self.external_mappings is not None:
            external_mappings.append(self.external_mappings)
        for item in self.registered_items.values():
            if isinstance(item, Subsystem):
                external_mappings.extend(item._get_external_mappings())

        return external_mappings

    def get_run_after_solve(self):
        run_after_solve = [getattr(self, name) for name in self.run_after_solve]

        for item in self.registered_items.values():
            if isinstance(item, Subsystem):
                run_after_solve.extend(item.get_run_after_solve())

        return run_after_solve

    def get_post_step(self):
        post_step = List.empty_list(types.FunctionType(types.void()))
        for x in [getattr(self, name) for name in self.post_step]:
            post_step.append(x)

        for item in self.registered_items.values():
            if isinstance(item, Subsystem):
                for x in item.get_post_step():
                    post_step.append(x)

        return post_step

    def get_external_mappings(self):
        external_mappings = self._get_external_mappings()

        for external_mapping in external_mappings:
            for mapping in external_mapping.external_mappings:
                for alias in mapping.dataframe_aliases:
                    if not self.find_variable(alias):
                        raise ValueError(f"No variable named '{alias}' in system '{self.tag}' "
                                         f"(could not map external data")

        return external_mappings


class ItemSet(Subsystem, EquationBase):

    def __init__(self, set_structure, tag):

        tag = "SET_" + tag
        super().__init__(tag)
        self.items = []
        set_structure_flat = set_structure

        self.item_ids = []

        self.item_type = None

        for item in set_structure_flat:
            item.part_of_set = True
            if not self.item_type:
                self.item_type = type(item)

            if not isinstance(item, self.item_type):
                raise TypeError(f'Error in registering set {tag} - '
                                f'All items in a set must have same type! This set is of type {self.item_type} '
                                f'not {type(item)}!')

            self.items.append(item)
            self.item_ids.append(item.id)
            if item.parent_set is None:
                item.parent_set = tag
            else:
                raise ValueError(f'Item {item} already part of set {item.parent_set} - cannot add to {tag}')

        tag_count = 0

        for item in set_structure_flat:
            for ns in item.registered_namespaces:
                tag_ = ns.tag
                if not (tag_ in self.registered_namespaces.keys()):
                    sns = SetNamespace(self, tag_, self.item_ids)
                    sns.add_equations(list(ns.associated_equations.values()), False, create_variables=False,
                                      set_equation=True)
                    sns.add_item(item, ns)
                    ns.clear_equations()
                    self.register_namespace(sns)
                else:
                    self.registered_namespaces[tag_].add_item(item, ns)
                if not ns.part_of_set:
                    ns.part_of_set = sns
                else:
                    ValueError(f'namespace {ns} already in set {ns.part_of_set}')
                tag_count += 1
        for ns in self.registered_namespaces.values():
            ns.items = self.items
        self.register_items(set_structure_flat)
