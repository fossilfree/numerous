from numerous.declarative.specification import ItemsSpec, ItemNotAssignedError
from numerous.declarative.watcher import watcher

import pytest

from tests.test_declarative.mock_objects import TestItemSpec, TestModule

def test_clone():
    test_spec = TestItemSpec()
    clone = test_spec._clone()

    assert clone != test_spec, 'Clone should create another ItemSpec'

def test_check_assigned():
    class TestItems(ItemsSpec):
        side1 = TestModule

    items = TestItems()

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    items.side1 = None
    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    items.side1 = TestModule

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    items.side1 = object()

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    items.side1 = TestModule(tag='test')

    items._check_assigned()

    class TestItems2(ItemsSpec):
        side1: TestModule

    items = TestItems2()

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    items.side1 = None
    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    items.side1 = TestModule

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    items.side1 = object()

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    items.side1 = TestModule(tag='test')

    items._check_assigned()

def test_finalize():

    class TestItems(ItemsSpec):
        side1: TestModule

    items = TestItems()
    watcher.add_watched_object(items)

    items.side1 = TestModule()

    items.finalize()

    assert items.side1._finalized, "Side 1 should be finalized"