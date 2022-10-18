from ..engine_ext import ScopeSpec, Parameter, ModuleSpec, Module, ItemsSpec, ItemNotAssignedError, create_mappings, MappingTypes, EquationSpec
from typing import Annotated
import pytest

class TestSpec(ScopeSpec):
    A = Parameter(0)

class ExtTestSpec(TestSpec):
    B = Parameter(0)

class TestModule(Module):
    """
    Class implementing a test module
    """

    tag: Annotated[str, "tag for the model"] = 'volume'

    default = TestSpec()
    
    def __init__(self, tag=None):
        super(TestModule, self).__init__(tag)




class TestModuleWithItems(Module):
    """
    Class implementing a test module
    """

    tag: Annotated[str, "tag for the model"] = 'testmodwithitems'

    class TestModuleItems(ItemsSpec):
        A: Annotated[TestModule, "A is assigned to a test module"]
        B: TestModule

    items = TestModuleItems()

    def __init__(self, A: TestModule, B: TestModule, tag=None):

        super(TestModuleWithItems, self).__init__(tag)

        self.items.A = A
        self.items.B = B


def test_ScopeSpec():


    test_spec = TestSpec()
    clone = test_spec.clone()

    assert clone != test_spec,  'Clone should create another TestSpec'

def test_extended_ScopeSpec():


    test_spec = ExtTestSpec()

    assert hasattr(test_spec, "A")
    assert hasattr(test_spec, "B")

    assert "B" in test_spec._variables

    # A should have been inherited!
    assert "A" in test_spec._variables, "A should have been inherited!"


def test_ModuleSpec():


    module_spec = ModuleSpec(TestModule)
    assert module_spec._namespaces['default'] != TestModule.default
    assert module_spec._namespaces['default'] == module_spec.default
    assert module_spec.default != TestModule.default

def test_ItemSpec():

    class TestItems(ItemsSpec):
        side1 = TestModule

    items = TestItems()

    with pytest.raises(ItemNotAssignedError):
        items.check_assigned()

    items.side1 = None
    with pytest.raises(ItemNotAssignedError):
        items.check_assigned()

    items.side1 = TestModule

    with pytest.raises(ItemNotAssignedError):
        items.check_assigned()

    items.side1 = object()

    with pytest.raises(ItemNotAssignedError):
        items.check_assigned()

    items.side1 = TestModule(tag='test')

    items.check_assigned()

def test_assign_mapping():
    class TestItems(ItemsSpec):
        side1 = TestModule
        side2 = TestModule

    items = TestItems()
    assert items.side1 != items.side2
    assert items.side1.default != items.side2.default
    assert items.side1.default.A != items.side2.default.A

    with create_mappings() as mappings:
        items.side1.default.A = items.side2.default.A

    assert items.side1.default.A != items.side2.default.A

    assert len(mappings.mappings) == 1, 'Only made 1 mapping'

    assert mappings.mappings[0][0] == items.side1.default.A, "side 1 A not mapped correctly"
    assert mappings.mappings[0][1] == items.side2.default.A, "side 2 A not mapped correctly"
    assert mappings.mappings[0][2] == MappingTypes.ASSIGN

def test_assign_hint_mapping():
    class TestItems(ItemsSpec):
        side1: TestModule
        side2: TestModule

    items = TestItems()

    assert items.side1.default != items.side2.default
    assert items.side1.default.A != items.side2.default.A

    with create_mappings() as mappings:
        items.side1.default.A = items.side2.default.A

    assert items.side1.default.A != items.side2.default.A

    assert len(mappings.mappings) == 1, 'Only made 1 mapping'

    assert mappings.mappings[0][0] == items.side1.default.A, "side 1 A not mapped correctly"
    assert mappings.mappings[0][1] == items.side2.default.A, "side 2 A not mapped correctly"
    assert mappings.mappings[0][2] == MappingTypes.ASSIGN

def test_add_mapping():
    class TestItems(ItemsSpec):
        side1 = TestModule
        side2 = TestModule

    items = TestItems()
    assert items.side1 != items.side2
    assert items.side1.default != items.side2.default
    assert items.side1.default.A != items.side2.default.A

    with create_mappings() as mappings:
        items.side1.default.A += items.side2.default.A

    assert items.side1.default.A != items.side2.default.A

    assert len(mappings.mappings) == 1, 'Only made 1 mapping'

    assert mappings.mappings[0][2] == MappingTypes.ADD
    assert mappings.mappings[0][0] == items.side1.default.A, "side 1 A not mapped correctly"
    assert mappings.mappings[0][1] == items.side2.default.A, "side 2 A not mapped correctly"

def test_module_with_itemspec():
    A = TestModule(tag="A")
    B = TestModule(tag="B")
    with pytest.raises(ItemNotAssignedError):

        tm = TestModuleWithItems(A=A, B=None)
        tm.finalize()

    tm = TestModuleWithItems(A=A, B=B)
    tm.finalize()

class TestModuleWithEquation(Module):
    """
    Class implementing a test module
    """

    tag: Annotated[str, "tag for the model"] = 'volume'

    default = TestSpec()

    @EquationSpec(default)
    def eq(self, scope:TestSpec):
        ...

class ExtendingTestModuleWithEquation(TestModuleWithEquation):
    ...
    #default = TestSpec()

    #@EquationSpec(default)
    #def eq2(self, scope: TestSpec):
    #    ...

class TestSpec2(TestSpec):
    ...

class ExtendingTestModuleWithEquationOverwrite(TestModuleWithEquation):



    default = TestSpec2()

    @EquationSpec(default)
    def eq2(self, scope: TestSpec):
        ...

class ExtendingTestModuleWithEquationOverwriteReuse(TestModuleWithEquation):


    default = TestSpec()

    @EquationSpec(default)
    def eq2(self, scope: TestSpec):
        ...

def test_module_extending_another():

    e = ExtendingTestModuleWithEquation()
    e.finalize()
    assert isinstance(e._scopes['default'], TestSpec)

def test_module_extending_and_overwriting_another():

    e = ExtendingTestModuleWithEquationOverwrite()
    e.finalize()
    assert isinstance(e._scopes['default'], TestSpec2)

    assert ExtendingTestModuleWithEquationOverwrite.eq2 in e._scopes['default']._equations
    assert ExtendingTestModuleWithEquationOverwrite.eq in e._scopes['default']._equations

    e = ExtendingTestModuleWithEquationOverwriteReuse()
    e.finalize()
    assert isinstance(e._scopes['default'], TestSpec)

    assert ExtendingTestModuleWithEquationOverwriteReuse.eq2 in e._scopes['default']._equations
    assert ExtendingTestModuleWithEquationOverwriteReuse.eq in e._scopes['default']._equations


