from numerous.declarative.specification import ItemsSpec, ItemNotAssignedError


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

    """
    with pytest.raises(ItemNotAssignedError):
        items.side1 = None
        items._check_assigned()

    items.side1 = TestModule

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    items.side1 = object()

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()
    """

    items.side1 = TestModule(tag='test')

    items._check_assigned()

    class TestItems2(ItemsSpec):
        side1: TestModule

    items = TestItems2()

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()

    """
    with pytest.raises(ItemNotAssignedError):
        items.side1 = None
        items._check_assigned()

    items.side1 = TestModule

    with pytest.raises(ItemNotAssignedError):
        items._check_assigned()
    """

    with pytest.raises(TypeError):
        items.side1 = object()

    items.side1 = TestModule(tag='test')

    items._check_assigned()

def test_finalize():

    class TestItems(ItemsSpec):
        side1: TestModule

    items = TestItems()

    items.side1 = TestModule()

    items.finalize()

    assert items.side1._finalized, "Side 1 should be finalized"