from numerous.declarative.watcher import watcher

from numerous.declarative.specification import ScopeSpec
from numerous.declarative.variables import Parameter


class TestSpec(ScopeSpec):
    A = Parameter(0)

import pytest

def test_watcher():
    watcher.declarations = []

    ts = TestSpec()

    assert watcher.declarations.index(ts) >= 0, "ts is not in watchers list"

    ts.attach()

    watcher.finalize()
