from numerous.declarative.watcher import watcher

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
    print("!")
    items = TestItemSpec()
    print("!!")

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

def test_watch_module():
    tm = TestModule()

    tm.finalize()

def test_watch_itemsspec_clone():

    tis = TestItemSpec()
    watcher.add_watched_object(tis)
    tis_clone = tis._clone()
    tis.finalize()
    watcher.add_watched_object(tis_clone)
    tis_clone.finalize()
    watcher.finalize()


