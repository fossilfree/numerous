from numerous.declarative.specification import ScopeSpec, Module, ItemsSpec
from numerous.declarative.variables import Parameter


class TestSpec(ScopeSpec):
    A = Parameter(0)


class ExtTestSpec(TestSpec):
    B = Parameter(0)


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
