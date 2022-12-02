from numerous.declarative import ItemsSpec, Module, ModuleSpec


import pytest

@pytest.fixture
def TestItems():
    class Items(ItemsSpec):

        a: Module

    return Items

def test_scope_spec(TestItems):

    items = TestItems()

    assert "a" in items._references
    assert isinstance(items.a, ModuleSpec)