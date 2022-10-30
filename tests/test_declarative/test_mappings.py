from numerous.declarative.specification import create_mappings, ItemsSpec, MappingOutsideMappingContextError
from .test_ModuleSpec import TestModule
import pytest


class TestItems(ItemsSpec):
    side1: TestModule
    side2: TestModule

def test_mappings():

    items = TestItems()
    items.side1 = TestModule()
    items.side2 = TestModule()

    items.side1.default.A = items.side2.default.A

