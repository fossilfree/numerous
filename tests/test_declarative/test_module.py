from numerous.declarative.specification import create_mappings, ItemsSpec, MappingOutsideMappingContextError, Module, ScopeSpec, EquationSpec
from numerous.declarative.variables import Parameter
from numerous.engine import model, simulation

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

    class TestSys(Module):
        tag = "test sys"

        class TestItems(ItemsSpec):
            A: TestModule
            B: TestModule
            tm: TestModuleWithItems

        items = TestItems()

        def __init__(self):
            super(TestSys, self).__init__(self.tag)
            self.items.A = TestModule(tag='tm1')
            self.items.B = TestModule(tag='tm2')
            self.items.tm = TestModuleWithItems(A=self.items.A, B=self.items.B)

    ts = TestSys()
    ts.finalize()

    m = model.Model(ts)

    # Define simulation
    s = simulation.Simulation(
        m,
        t_start=0, t_stop=10, num=10, num_inner=1, max_step=10
    )
    # Solve
    s.solve()

    def last(var):
        return s.model.historian_df[var.path.primary_path].tail(1).values[0]

    assert last(ts.items.A.default.var1) == pytest.approx(19.0)