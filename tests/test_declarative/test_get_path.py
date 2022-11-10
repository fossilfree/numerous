from numerous.declarative.specification import ItemsSpec, Module, ScopeSpec, EquationSpec, \
    recursive_get_attr
from numerous.declarative.mappings import create_mappings
from numerous.declarative.exceptions import MappingOutsideMappingContextError, NotMappedError, FixedMappedError
from numerous.declarative.variables import Parameter, Constant

import pytest
from typing import Annotated

class TestLeafModule(Module):
    """
    Class implementing a test module
    """
    class Variables(ScopeSpec):
        a = Parameter(0)

    default = Variables()

    tag: str = 'mod'

    def __init__(self, tag=None):
        super(TestLeafModule, self).__init__(tag)




class TestBranchLevel1Module(Module):
    """
    Class implementing a test module
    """

    class Items(ItemsSpec):
        branch: TestLeafModule

    tag: str = 'mod1'

    items = Items()

class TestBranchLevel2Module(Module):
    """
    Class implementing a test module
    """

    class Items(ItemsSpec):
        branch: TestBranchLevel1Module

    tag: str = 'mod2'

    items = Items()

class TestBranchLevel3Module(Module):
    """
    Class implementing a test module
    """

    class Items(ItemsSpec):
        branch: TestBranchLevel2Module

    tag: str = 'mod3'

    items = Items()

    def __init__(self, tag=None):


        super(TestBranchLevel3Module, self).__init__(tag)


def test_get_path():
    tm = TestBranchLevel3Module()

    path = tm.items.branch.items.branch.items.branch.default.a.get_path(tm)
    print(path)

    var = recursive_get_attr(tm, path)

    assert var == tm.items.branch.items.branch.items.branch.default.a
