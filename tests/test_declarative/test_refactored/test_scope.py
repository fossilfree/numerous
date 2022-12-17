from numerous.declarative import ScopeSpec, ItemsSpec, Module, equation
from numerous.declarative.variables import Parameter


import pytest

@pytest.fixture
def TestScope():
    class TestScope(ScopeSpec):

        a = Parameter(0)

    return TestScope


def test_mapping(TestScope):

    test_scope = TestScope()
    test_scope_2 = TestScope()
    test_scope_2.a.add_assign_mapping(test_scope.a)

    assert test_scope is not test_scope_2
    assert test_scope.a is not test_scope_2.a
    assert len(test_scope_2.a.mappings) == 1
    assert test_scope.a is test_scope_2.a.mappings[0][1]


def test_mapping_sum(TestScope):
    test_scope = TestScope()
    test_scope_2 = TestScope()


    test_scope_2.a.add_assign_mapping(test_scope.a)

    assert len(test_scope_2.a.mappings) == 1
    assert test_scope.a is test_scope_2.a.mappings[0][1]


