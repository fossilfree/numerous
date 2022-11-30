from numerous.declarative.clonable_interfaces import ClassVarSpec, Clonable, ParentReference, clone_recursive, clone_references

import pytest

@pytest.fixture
def class_var_spec():

    class Variable(Clonable):
        ...

    class VarSpec(ClassVarSpec):

        def __init__(self):
            super(VarSpec, self).__init__(class_var_type=Variable)

    class Variables(VarSpec):
        a = Variable()

    return Variables

def test_class_var_captured(class_var_spec):

    var = class_var_spec()
    var2 = class_var_spec()

    assert "a" in var._references
    assert var.a != class_var_spec.a
    assert var.a != var2.a