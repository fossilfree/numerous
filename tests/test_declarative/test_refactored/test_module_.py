from numerous.declarative import Module, ScopeSpec, Variable, Connector, ItemsSpec, equation, get_value_for, set_value_from
from numerous.declarative.mappings import create_mappings
import pytest

@pytest.fixture
def TestModule():
    class TestModule(Module):
        class Variables(ScopeSpec):
            a = Variable()

        variables = Variables()

        connector = Connector(a=get_value_for(variables.a))
        ...

    return TestModule


def test_common_ref_space(TestModule):

    mod1 = TestModule()

    assert mod1.connector.channels['a'][0] is mod1.variables.a

    mod2 = TestModule()

    assert mod1.connector.a is not mod2.connector.a
    assert mod1.variables.a is not mod2.variables.a


def test_module():
    class TestSub(Module):
        class Variables(ScopeSpec):
            b = Variable()

        variables = Variables()

        connector = Connector(b=get_value_for(variables.b))

        def __init__(self):
            super(TestSub, self).__init__()

        @equation(variables)
        def eq(self, scope: Variables):
            scope.b = 1


    class Test(Module):
        class Variables(ScopeSpec):
            a = Variable()

        variables = Variables()

        class Modules(ItemsSpec):
            mod1: TestSub
            mod2: TestSub

        modules = Modules()


        with create_mappings() as mappings:
            variables.a.add_assign_mapping(modules.mod1.variables.b)
            modules.mod1.variables.b.add_sum_mapping(modules.mod2.variables.b)

        connector = Connector(a=get_value_for(variables.a))

        def __init__(self):
            super(Test, self).__init__()

            #self.modules.mod1 = TestSub()
            ...


    test = Test()

    assert test.connector.channels['a'][0] is test.variables.a

    test2 = Test()

    assert test.variables.a is not test2.variables.a
    assert test.modules.mod1 is not test2.modules.mod1

    assert test.modules.mod1.connector.channels['b'] is not test2.modules.mod1.connector.channels['b']

    #assert test.variables.a.mappings[0][1] is test.modules.mod1.variables.b
    #assert test2.variables.a.mappings[0][1] is not test.variables.a.mappings[0][1]

    assert test.modules.mod1.variables.b is not test.modules.mod2.variables.b

    #assert test.modules.mod1.variables.b.mappings[0][1] is test.modules.mod2.variables.b


