from numerous.declarative.specification import ScopeSpec, ItemsSpec, Module
from numerous.declarative.variables import Parameter


class TestSpec(ScopeSpec):
    A = Parameter(0)


class TestItemSpec(ItemsSpec):
    ...


class TestModule(Module):
    """
    Class implementing a test module
    """
    print('!')
    tag: str = 'mod'

    # default = TestSpec()
    print('!!')

    # items = TestItemSpec()

    def __init__(self, tag=None):
        super(TestModule, self).__init__(tag)

    # @EquationSpec(default)
    def eval(self, scope: TestSpec):
        scope.var1 = 19


def test_mod():
    ...
    TestModule()
