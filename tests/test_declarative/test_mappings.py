from numerous.declarative.specification import create_mappings, ItemsSpec, MappingOutsideMappingContextError, Module, ScopeSpec
from numerous.declarative.variables import Parameter
from .mock_objects import TestModule
import pytest

class TestModuleWithItems(Module):
    class Variables(ScopeSpec):
        a = Parameter(0)

    default = Variables()
    class Items(ItemsSpec):
        side1: TestModule
        side2: TestModule

    items = Items()

    with create_mappings() as mappings:
        default.a = items.side1.default.A

    def __init__(self, tag):
        super(TestModuleWithItems, self).__init__(tag)

        self.items.side1 = TestModule("tm1")
        self.items.side2 = TestModule("tm2")

def test_mappings():

    twi = TestModuleWithItems("sys")
    twi.finalize()
    assert len(twi.mappings.mappings) == 1

    ...

