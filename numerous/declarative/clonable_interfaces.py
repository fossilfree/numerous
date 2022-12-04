from abc import ABC, abstractmethod
from uuid import uuid4
import inspect

class ParentReference:
    parent: object
    attr: str

    def __init__(self, parent: object, attr: str):
        self.parent = parent
        self.attr = attr

    def __repr__(self):
        return f"{self.parent}, {self.attr}, {super(ParentReference, self).__repr__()}"


def clone_recursive(references: dict, cloned_references_global: dict = None):
    if cloned_references_global is None:
        cloned_references_global = {}

    cloned_references = {}

    for key, reference in references.items():

        if not reference._id in cloned_references_global:
            cloned_references_global[reference._id] = reference.clone_with_references(cloned_references_global)

        cloned_references[key] = cloned_references_global[reference._id]

    return cloned_references


def clone_references(references, cloned_references_global=None):
    if cloned_references_global is None:
        cloned_references_global = {}

    cloned_references = clone_recursive(references, cloned_references_global=cloned_references_global)


    return cloned_references


class Clonable(ABC):
    _references: dict
    _parent: ParentReference = None
    _bind: bool = False
    _set_parent_on_references: bool = False
    _id: str
    _initialized: bool = False
    _clone_of: object
    _first_clone: object
    #_clone_refs: bool

    def __init__(self, bind=False,
                 set_parent_on_references=False, clone_refs=True):

        self._id = str(uuid4())
        self._clone_of = None
        self._first_clone = None
        self._references = {}

        self._bind = bind
        self._set_parent_on_references = set_parent_on_references

        self._initialized = True

    def configure_clone(self, clone, references: dict = None, do_clone=True):
        clone._bind = self._bind
        clone._set_parent_on_references = self._set_parent_on_references
        #clone._clone_refs = self._clone_refs

        if do_clone:
            references = clone_references(references)

        clone.set_references(references)

    def clone_with_references(self, cloned_references_global: dict = None):

        cloned_references = clone_recursive(self._references, cloned_references_global=cloned_references_global)

        clone_ = self.clone()
        self.configure_clone(clone_, cloned_references, do_clone=False)
        clone_._parent = self._parent
        return clone_

    def add_reference(self, key, reference):

        # clone_ = reference.clone_references()

        self._references[key] = reference

        if self._bind:

            self._initialized = False
            self.__setattr__(key, reference, add_ref=True)
            self._initialized = True

            if self._set_parent_on_references:

                reference.set_parent(ParentReference(self, key))

        elif self._set_parent_on_references:
            raise KeyError("Can only set parent if reference is set to bind.")

    def __setattr__(self, key, value, add_ref=False):
        super(Clonable, self).__setattr__(key, value)
    def set_references(self, references):

        for key, reference in references.items():
            self.add_reference(key, reference)

    def set_parent(self, parent_ref: ParentReference):
        self._parent = parent_ref

    def get_path(self, host):


        if not self._parent and not self._clone_of:
            if not self._first_clone:
                print('here')
                print(self)
                print(host)
                ...
            self._parent = self._first_clone._parent

        path = [self._parent.attr]

        if not self._parent.parent == host:
            print(path)
            path = self._parent.parent.get_path(host) + path

        return path


    def clone(self):
        clone = self.__class__()
        clone._clone_of = self
        self.configure_clone(clone, self._references)
        if not self._first_clone:
            self._first_clone = clone

        return clone

    def get_references_of_type(self, type):
        return {k: v for k, v in self._references.items() if isinstance(v, type)}

def get_class_vars(obj, class_var_type:type, _handle_annotations=None):
    """

    """

    vars = {}
    class_var = {}


    if inspect.isclass(obj):
        _class = obj
        _is_obj = False
    else:
        _class = obj.__class__
        _is_obj = True


    for b in _class.__bases__:
        class_var.update(b.__dict__)

    class_var.update(_class.__dict__)

    class_var.update(obj.__dict__)


    for key, var in class_var.items():
        if isinstance(var, class_var_type):
            vars[key] = var

    if _handle_annotations:
        annotations = _class.__annotations__

        for var, hint in annotations.items():

            if (_is_obj and not hasattr(obj, var)) or not _is_obj:
                vars[var] = _handle_annotations(hint)

    return vars


class ClassVarSpec(Clonable):

    _class_var_type: type = None
    _handle_annotations: object = None

    def __init__(self, clone_refs=True, set_parent_on_references=True, class_var_type=None, handle_annotations=None):
        self._class_var_type = class_var_type
        self._handle_annotations = handle_annotations

        super(ClassVarSpec, self).__init__(bind=True, set_parent_on_references=set_parent_on_references, clone_refs=clone_refs)


        if self._class_var_type is not None:

            references = get_class_vars(self, self._class_var_type, _handle_annotations=self._handle_annotations)

            cloned_references = clone_references(references)

            self.set_references(cloned_references)
