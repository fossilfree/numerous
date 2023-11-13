from numerous.declarative.specification import ModuleSpec
from .mock_objects import TestModule


def test_ModuleSpec():
    module_spec = ModuleSpec(TestModule)

    assert module_spec._namespaces['default'] != TestModule.default
    assert module_spec._namespaces['default'] == module_spec.default
    assert module_spec.default != TestModule.default

    assert module_spec._item_specs['items'] != TestModule.items
    assert module_spec._item_specs['items'] == module_spec.items
    assert module_spec.items != TestModule.items
