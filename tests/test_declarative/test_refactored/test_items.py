from numerous.declarative import ItemsSpec, Module
from numerous.declarative.module import ModuleSpec


import pytest

@pytest.fixture
def TestItems():
    class Items(ItemsSpec):

        a: Module

    return Items

def test_scope_spec(TestItems):

    items = TestItems()

    assert "a" in items.__dict__
    assert isinstance(items.a, ModuleSpec)