from numerous.declarative.clonable_interfaces import Clonable, ParentReference

import pytest

class ScopeSpec(Clonable):

    def __init__(self, variables: dict):

        super(ScopeSpec, self).__init__(clone_refs=True)
        self.set_references(variables)


class ModuleSpec(ClassVarSpec):

    def __init__(self, items_specs: dict=None):

        super(ModuleSpec, self).__init__(clone_refs=True)
        self.set_references(items_specs)


class ItemsSpec(ClassVarSpec):

    def __init__(self, module_specs: dict = None):

        super(ItemsSpec, self).__init__(clone_refs=True)
        self.set_references(references=module_specs)


class Connector(ClassVarSpec):

    def __init__(self, channels: dict = None):

        super(Connector, self).__init__(clone_refs=True, set_parent_on_references=False)
        self.set_references(references=channels)


def test_clonable():

    module_spec_level0 = ModuleSpec(items_specs={})

    items_spec_level0 = ItemsSpec(module_specs={'mod0': module_spec_level0})

    connector = Connector(channels={'a': items_spec_level0.mod0})

    module_spec_level1 = ModuleSpec(items_specs={'items0': items_spec_level0, 'connector1': connector})
    module_spec_level1_2 = ModuleSpec(items_specs={'items0': items_spec_level0, 'connector1': connector})


    assert connector.a == items_spec_level0.mod0
    assert items_spec_level0._references['mod0'] == items_spec_level0.mod0



    assert module_spec_level1_2.connector1.a == module_spec_level1_2.items0.mod0


    items_spec_level1 = ItemsSpec(module_specs={'mod1': module_spec_level1, 'mod2': module_spec_level1_2})

    module_spec_level2 = ModuleSpec(items_specs={'items': items_spec_level1})


    a1 = module_spec_level2.items.mod1.connector1.a
    a2 = module_spec_level2.items.mod2.connector1.a

    assert a1.get_path(module_spec_level2) == ['items', 'mod1', 'items0', 'mod0']
    assert a2.get_path(module_spec_level2) == ['items', 'mod2', 'items0', 'mod0']
