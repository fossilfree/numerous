from numerous.declarative.clonable_interfaces import Clonable, ParentReference, clone_recursive, clone_references

import pytest

@pytest.fixture
def cloneable():
    return Clonable(bind=True, set_parent_on_references=True)

@pytest.fixture
def another_cloneable():
    return Clonable(bind=True, set_parent_on_references=True)

@pytest.fixture
def non_parent_cloneable():
    return Clonable(bind=True, set_parent_on_references=False)

@pytest.fixture
def yet_another_cloneable():
    return Clonable(bind=True)

# Expect
# Creating a cloneable should create a new instance with clones of the references

def test_clone(cloneable):

    clone = cloneable.clone()

    assert clone != cloneable

def test_clone_recursive(cloneable):
    global_reference_dict = {}
    ref_key = 'cloneable'
    ref_key2 = 'cloneable2'

    references = {ref_key: cloneable, ref_key2: cloneable}
    cloned_references = clone_recursive(references, global_reference_dict)

    assert cloneable._id in global_reference_dict
    assert ref_key in cloned_references
    assert cloned_references[ref_key] != cloneable
    assert cloned_references[ref_key] == cloned_references[ref_key2]


def test_clone_references(cloneable, another_cloneable, yet_another_cloneable):

    cloneable.add_reference('yet_another_cloneable', yet_another_cloneable)
    another_cloneable.add_reference('yet_another_cloneable', yet_another_cloneable)

    cloned_references = clone_references({"key1": cloneable, "key2": another_cloneable})

    assert cloned_references['key1'].yet_another_cloneable == cloned_references['key2'].yet_another_cloneable


def test_clone_reference(cloneable, another_cloneable):

    cloneable.add_reference("another", another_cloneable)

    assert hasattr(cloneable, "another")

    clone = cloneable.clone()

    assert clone.another != another_cloneable

def test_clone_deep_reference(cloneable, another_cloneable, yet_another_cloneable):

    ref_key_another = "another"
    ref_key_yet_another = "yet_another"

    another_cloneable.add_reference(ref_key_yet_another, yet_another_cloneable)
    cloneable.add_reference(ref_key_another, another_cloneable)

    assert hasattr(cloneable, ref_key_another)
    assert hasattr(another_cloneable, ref_key_yet_another)
    assert hasattr(cloneable.another, ref_key_yet_another)

    clone = cloneable.clone()

    assert clone.another != another_cloneable
    assert clone.another.yet_another != yet_another_cloneable


def test_parent(cloneable, another_cloneable):

    ref_key_another = "another"
    ref_key_another2 = "another2"

    cloneable.add_reference(ref_key_another, another_cloneable)
    cloneable.add_reference(ref_key_another2, another_cloneable.clone())

    assert hasattr(cloneable, ref_key_another)
    assert hasattr(cloneable, ref_key_another2)

    clone = cloneable.clone()

    assert clone.another != cloneable.another2

    assert clone.another2._parent != clone.another._parent
    assert clone.another2._parent.attr != clone.another._parent.attr


    #assert clone.another2._parent.attr == ref_key_another2
    #assert clone.another._parent.attr == ref_key_another


def test_get_path(cloneable, another_cloneable, yet_another_cloneable, non_parent_cloneable):

    ref_key_another = "another"
    ref_key_another2 = "another2"
    ref_key_yet_another = "yet_another"
    ref_key_non_parent = "non_parent"

    another_cloneable.add_reference(ref_key_yet_another, yet_another_cloneable)

    non_parent_cloneable.add_reference(ref_key_yet_another, yet_another_cloneable)
    another_cloneable.add_reference(ref_key_non_parent, non_parent_cloneable)

    cloneable.add_reference(ref_key_another, another_cloneable)

    cloneable.add_reference(ref_key_another2, another_cloneable.clone())

    assert hasattr(cloneable, ref_key_another)
    assert hasattr(cloneable, ref_key_another2)

    assert hasattr(another_cloneable, ref_key_yet_another)
    assert hasattr(cloneable.another, ref_key_yet_another)
    assert hasattr(cloneable.another2, ref_key_yet_another)

    assert cloneable.another != cloneable.another2
    assert cloneable.another.yet_another != cloneable.another2.yet_another

    clone = cloneable.clone()

    assert clone.another != cloneable.another2
    assert clone.another.yet_another != cloneable.another2.yet_another

    assert clone.another2.yet_another._parent.parent == clone.another2
    assert clone.another.yet_another._parent.parent == clone.another

    assert clone.another2._parent != clone.another._parent

    assert clone.another.yet_another.get_path(clone) == [ref_key_another, ref_key_yet_another]

    assert clone.another.non_parent.yet_another == clone.another.yet_another
    assert clone.another.non_parent.yet_another.get_path(clone) == [ref_key_another, ref_key_yet_another]

    assert clone.another2.yet_another.get_path(clone) == [ref_key_another2, ref_key_yet_another]

def test_get_path_deep(cloneable, another_cloneable, yet_another_cloneable, non_parent_cloneable):

    ref_key_another = "another"
    ref_key_another2 = "another2"
    ref_key_yet_another = "yet_another"
    ref_key_non_parent = "non_parent"

    another_cloneable.add_reference(ref_key_yet_another, yet_another_cloneable)

    non_parent_cloneable.add_reference(ref_key_yet_another, yet_another_cloneable)

    cloneable.add_reference(ref_key_non_parent, non_parent_cloneable)

    cloneable.add_reference(ref_key_another, another_cloneable)

    cloneable.add_reference(ref_key_another2, another_cloneable.clone())

    clone = cloneable.clone()

    assert cloneable.another.yet_another != clone.another.yet_another

    assert clone.non_parent.yet_another == clone.another.yet_another
    assert clone.non_parent.yet_another.get_path(clone) == [ref_key_another, ref_key_yet_another]

    assert clone.another2.yet_another.get_path(clone) == [ref_key_another2, ref_key_yet_another]