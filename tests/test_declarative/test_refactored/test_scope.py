from numerous.declarative import ScopeSpec, ItemsSpec, Module, EquationSpec
from numerous.declarative.variables import Parameter
from numerous.declarative.mappings import create_mappings

import pytest

@pytest.fixture
def TestScope():
    class TestScope(ScopeSpec):

        a = Parameter(0)

    return TestScope

def test_scope_spec(TestScope):

    test_scope = TestScope()

    assert "a" in test_scope.get_variables()

def test_mapping(TestScope):

    test_scope = TestScope()
    test_scope_2 = TestScope()
    with create_mappings() as mappings:
        test_scope_2.a = test_scope.a

    assert len(test_scope_2.a.mappings) == 1
    assert test_scope.a._id in test_scope_2.a.mappings


def test_mapping_sum(TestScope):
    test_scope = TestScope()
    test_scope_2 = TestScope()

    with create_mappings() as mappings:
        test_scope_2.a += test_scope.a

    assert len(test_scope_2.a.mappings) == 1
    assert test_scope.a._id in test_scope_2.a.mappings


