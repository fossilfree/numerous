from numerous.declarative.specification import create_mappings, ItemsSpec, MappingOutsideMappingContextError, Module, ScopeSpec, EquationSpec, NotMappedError, FixedMappedError
from numerous.declarative.variables import Parameter, Constant

import pytest
from typing import Annotated

class TestSpec(ScopeSpec):
    var1 = Parameter(0)


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

def test_module():
    tm = TestModule()
    tm.finalize()

class TestModuleWithItems(Module):
    """
    Class implementing a test module
    """

    tag: Annotated[str, "tag for the model"] = 'testmodwithitems'

    class Items(ItemsSpec):
        A: Annotated[TestModule, "A is assigned to a test module"]
        B: TestModule

    items = Items()

    def __init__(self, A: TestModule, B: TestModule, tag=None):
        super(TestModuleWithItems, self).__init__(tag)

        self.items.A = A
        self.items.B = B
        #A.default.var1 = B.default.var1

def test_module_with_items():
    A = TestModule('A')
    B = TestModule('B')

    tmi = TestModuleWithItems(A=A, B=B, tag="tmi")
    tmi.finalize()

class TestSysMustMapped(Module):
    tag = "test sys"
    class Variables(ScopeSpec):
        var1 = Parameter(0, must_be_mapped=True)
        var2 = Constant(0)

    default = Variables()
    class TestItems(ItemsSpec):
        A: TestModule
        B: TestModule
        tm: TestModuleWithItems

    items = TestItems()

    def __init__(self):
        super(TestSysMustMapped, self).__init__(self.tag)
        self.items.A = TestModule(tag='tm1')
        self.items.B = TestModule(tag='tm2')
        self.items.tm = TestModuleWithItems(A=self.items.A, B=self.items.B)

        #self.default.var1 = self.items.A.default.var1
        self.items.B.default.var1 = self.default.var2

def test_module_with_items_must_mapped_light():

    ts = TestSysMustMapped()

    with pytest.raises(NotMappedError):
        ts.finalize()

class TestSysFixed(Module):
    tag = "test sys"
    class Variables(ScopeSpec):
        var1 = Parameter(0)
        var2 = Constant(0)

    default = Variables()
    class TestItems(ItemsSpec):
        A: TestModule
        B: TestModule
        tm: TestModuleWithItems

    items = TestItems()

    def __init__(self):
        super(TestSysFixed, self).__init__(self.tag)
        self.items.A = TestModule(tag='tm1')
        self.items.B = TestModule(tag='tm2')
        self.items.tm = TestModuleWithItems(A=self.items.A, B=self.items.B)

        #self.default.var1 = self.items.A.default.var1
        self.default.var2 = self.items.B.default.var1

def test_module_with_items_fixed_light():

    ts = TestSysFixed()

    with pytest.raises(FixedMappedError):
        ts.finalize()