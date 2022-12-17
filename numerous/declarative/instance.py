import inspect
import uuid

from .interfaces import ModuleSpecInterface


def get_class_vars(obj, class_var_type:tuple[type], _handle_annotations=None):
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

        #class_var = _class.merge(class_var, b.__dict__, class_var_type)
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


class Class:
    _is_instance:bool = False
    _from: list
    _context: dict[str:object]

    def __init__(self):
        self._id = str(uuid.uuid4())
        self._from = []
        self._context = None

    def instance(self, context):

        if self._id in context:
            return context[self._id]
        else:
            instance_ = self._instance_recursive(context)
            instance_._is_instance = True
            instance_._from = self._from + [self]
            context[self._id] = instance_
            instance_._context = context
            #instance_._id = self._id
            return instance_

    def _instance_recursive(self, context:dict):

        return self._instance()

    def _instance(self):
        return self.__class__()

    def _items_of_type(self, type_: type):
        ...
        return {key: item for key, item in self.__dict__.items() if isinstance(item, type_)}


