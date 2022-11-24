from numerous.declarative.specification import ItemsSpec, Module, ScopeSpec
from numerous.declarative.variables import Parameter


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


class TestAssignOutside(Module):
    """
        Class implementing a test module
        """

    tag: str = 'mod'

    class Items(ItemsSpec):
        mod1: TestModule

    items = Items()


def test_assign_outside_init():
    class OuterTest(Module):
        """
            Class implementing a test module
            """

        tag: str = 'mod_out'

        class Items(ItemsSpec):
            inner: TestAssignOutside
            testmod: TestModule

        items = Items()

        def __init__(self, tag):
            super(OuterTest, self).__init__(tag)
            self.items.testmod = TestModule("tm")
            self.items.inner = TestAssignOutside("test")
            self.items.inner.items.mod1 = self.items.testmod

        items = Items()

    test_outer = OuterTest("outer")
    test_outer.finalize()
