from tests.test_declarative.mock_objects import TestSpec, ExtTestSpec


def test_clone():
    test_spec = TestSpec()
    clone = test_spec._clone()

    assert clone != test_spec, 'Clone should create another TestSpec'

def test_extending():
    test_spec = ExtTestSpec()

    assert hasattr(test_spec, "A")
    assert hasattr(test_spec, "B")

    assert "B" in test_spec._variables

    # A should have been inherited!
    assert "A" in test_spec._variables, "A should have been inherited!"