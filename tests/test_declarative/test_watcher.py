from numerous.declarative.watcher import Watcher
from numerous.declarative.context_managers import _active_declarative
from numerous.declarative.specification import ScopeSpec, ItemsSpec, Module, EquationSpec
from numerous.declarative.variables import Parameter
import pytest

class TestSpec(ScopeSpec):
    A = Parameter(0)

class TestItemSpec(ItemsSpec):
    ...

class TestModule(Module):
    """
    Class implementing a test module
    """
    tag: str = 'mod'

    default = TestSpec()
    items = TestItemSpec()

    def __init__(self, tag=None):
        super(TestModule, self).__init__(tag)

    @EquationSpec(default)
    def eval(self, scope: TestSpec):
        scope.var1 = 19

"""
def test_watcher():
    watcher.declarations = []

    ts = TestSpec()

    assert watcher.declarations.index(ts) >= 0, "ts is not in watchers list"

    ti = TestItemSpec()

    ts.attach()
    ti.attach()

    watcher.finalize()"""

def test_no_active_context():
    assert not _active_declarative.is_active_manager_context_set()

def test_watch_module():

    tm = TestModule()

    tm.finalize()
    assert not _active_declarative.is_active_manager_context_set()


def test_watch_itemsspec_clone():

    tis = TestItemSpec()

    tis_clone = tis._clone()
    tis.finalize()

    tis_clone.finalize()


def test_capture_not_used():
    class TestModuleNotUsed(Module):
        """
        Class implementing a test module
        """
        tag: str = 'mod'

        TestSpec()
        items = TestItemSpec()

        def __init__(self, tag=None):
            super(TestModuleNotUsed, self).__init__(tag)






