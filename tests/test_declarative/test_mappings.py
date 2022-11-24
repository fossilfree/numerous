import pytest

from numerous.declarative.mappings import create_mappings
from numerous.declarative.specification import ItemsSpec, Module, ScopeSpec, EquationSpec
from numerous.declarative.variables import Parameter
from numerous.engine import model, simulation


class TestSpec(ScopeSpec):
    var1 = Parameter(0)


class ExtTestSpec(TestSpec):
    var2 = Parameter(0)


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


class TestModuleWithItems(Module):
    class Variables(ScopeSpec):
        a = Parameter(0)

    default = Variables()

    class Items(ItemsSpec):
        side1: TestModule
        side2: TestModule

    items = Items()

    with create_mappings() as mappings:
        default.a = items.side1.default.var1
        items.side1.default.var1 = items.side2.default.var1

    def __init__(self, tag):
        super(TestModuleWithItems, self).__init__(tag)

        self.items.side1 = TestModule("tm1")
        self.items.side2 = TestModule("tm2")


def test_mappings():
    twi = TestModuleWithItems("sys")
    twi.finalize()
    assert len(twi.mappings.mappings) == 2

    ...


class Variables(ScopeSpec):
    var1 = Parameter(0)


class ExtVariables(Variables):
    var2 = Parameter(0)
    var3 = Parameter(0)


class TestSubNode(Module):
    """
    Class implementing a test module
    """

    tag: str = 'node'

    default = ExtVariables()

    def __init__(self, tag=None):
        super(TestSubNode, self).__init__(tag)


class VariablesConnector(ScopeSpec):
    side1_var1 = Parameter(0)
    side2_var1 = Parameter(0)
    side1_var2 = Parameter(0)
    side2_var2 = Parameter(0)
    F = Parameter(0)


class TestSubConnector(Module):
    """
    Class implementing a test module
    """

    tag: str = 'subconn'

    default = VariablesConnector()

    class Items(ItemsSpec):
        side1: TestSubNode
        side2: TestSubNode

    items = Items()

    with create_mappings() as mappings:
        items.side1.default.var1 += default.side1_var1
        items.side2.default.var1 += default.side2_var1
        default.side1_var2 = items.side1.default.var2
        default.side2_var2 = items.side2.default.var2
        items.side1.default.var3 = items.side2.default.var3

    def __init__(self, side1: TestSubNode, side2: TestSubNode, tag=None):
        super(TestSubConnector, self).__init__(tag)

        self.items.side1 = side1
        self.items.side2 = side2

    @EquationSpec(default)
    def eval(self, scope: Variables):
        scope.side1_var1 = scope.F
        scope.side2_var1 = scope.F * 2


class TestCompositeNodeConnector(Module):
    tag: str = ""

    class Items(ItemsSpec):
        side1: TestSubNode
        side2: TestSubNode
        side3: TestSubNode

    items = Items()

    def __init__(self, tag, F):
        super(TestCompositeNodeConnector, self).__init__(tag)

        self.items.side1 = TestSubNode("sub_node_1")
        self.items.side2 = TestSubNode("sub_node_2")
        self.items.side3 = TestSubNode("sub_node_3")

        self.connector = TestSubConnector(tag="connector", side1=self.items.side1, side2=self.items.side2)

        self.connector.default.F.value = 1

        self.connector2 = TestSubConnector(tag="connector2", side1=self.items.side2, side2=self.items.side3)

        self.connector2.default.F.value = F

        self.items.side1.default.var2.value = 10
        self.items.side2.default.var2.value = 20
        self.items.side2.default.var3.value = 30


def test_composite_connector():
    F = 1
    composite = TestCompositeNodeConnector("composite", F=F)
    composite.finalize()

    m = model.Model(composite, use_llvm=False)

    # Define simulation
    s = simulation.Simulation(
        m,
        t_start=0, t_stop=10, num=10, num_inner=1, max_step=10
    )
    # Solve
    s.solve()

    def last(var):
        return s.model.historian_df[var.path.primary_path].tail(1).values[0]

    assert last(composite.connector.default.side1_var1) == pytest.approx(F)
    assert last(composite.connector.default.side2_var1) == pytest.approx(2 * F)
    assert last(composite.connector2.default.side1_var1) == pytest.approx(F)
    assert last(composite.connector2.default.side2_var1) == pytest.approx(2 * F)

    assert last(composite.connector.default.side1_var2) == pytest.approx(last(composite.items.side1.default.var2))
    assert last(composite.connector2.default.side1_var2) == pytest.approx(last(composite.items.side2.default.var2))

    assert last(composite.items.side1.default.var1) == pytest.approx(last(composite.connector.default.side1_var1))
    assert last(composite.items.side2.default.var1) == pytest.approx(
        last(composite.connector.default.side2_var1) + last(composite.connector2.default.side1_var1))
    assert last(composite.items.side3.default.var1) == pytest.approx(last(composite.connector2.default.side2_var1))

    assert last(composite.items.side1.default.var2) == pytest.approx(10)
    assert last(composite.items.side2.default.var2) == pytest.approx(20)

    assert last(composite.items.side2.default.var2) == pytest.approx(20)


class TestCompositeNodeConnector(Module):
    tag: str = ""

    class Items(ItemsSpec):
        side1: TestSubNode
        side2: TestSubNode
        side3: TestSubNode

    items = Items()

    def __init__(self, tag, F):
        super(TestCompositeNodeConnector, self).__init__(tag)

        self.items.side1 = TestSubNode("sub_node_1")
        self.items.side2 = TestSubNode("sub_node_2")
        self.items.side3 = TestSubNode("sub_node_3")

        self.connector = TestSubConnector(tag="connector", side1=self.items.side1, side2=self.items.side2)

        self.connector.default.F.value = 1

        self.connector2 = TestSubConnector(tag="connector2", side1=self.items.side2, side2=self.items.side3)

        self.connector2.default.F.value = F

        self.items.side1.default.var2.value = 10
        self.items.side2.default.var2.value = 20
        self.items.side2.default.var3.value = 30
