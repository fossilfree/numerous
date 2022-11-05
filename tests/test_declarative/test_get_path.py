from numerous.declarative.specification import create_mappings, ItemsSpec, MappingOutsideMappingContextError, Module, ScopeSpec, EquationSpec, NotMappedError, FixedMappedError, recursive_get_attr
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

    path = tm.items.branch.items.branch.items.branch.default.a.get_rel_path(tm.__class__)
    print(path)

    var = recursive_get_attr(tm, path)

    assert var == tm.items.branch.items.branch.items.branch.default.a
