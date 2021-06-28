import ast
from enum import unique, IntEnum


class Imports:
    def __init__(self):
        self.as_imports = []
        self.from_imports = []

    def add_as_import(self, module_name, alias):
        self.as_imports.append((module_name, alias))

    def add_from_import(self, module_name, element):
        self.from_imports.append((module_name, element))


@unique
class EventTypes(IntEnum):
    TIME_EVENT = 0
    STATE_EVENT = 1


# TODO add setVAR/GridVar?
@unique
class NodeTypes(IntEnum):
    OP = 0
    ASSIGN = 1
    VAR = 2
    EQUATION = 3
    SUM = 4
    TMP = 5
    ASSIGNTUPLE = 6
    IF = 7

def recurse_Attribute(attr, sep='.'):
        if hasattr(attr, 'id'):
            return attr.id
        elif isinstance(attr.value, ast.Name):
            return attr.value.id + sep + attr.attr
        elif isinstance(attr.value, ast.Attribute):
            return recurse_Attribute(attr.value) + sep + attr.attr


def wrap_function(name, body, args, decorators):
    f = ast.FunctionDef(name)
    f.body = body
    f.decorator_list = decorators
    f.args = args
    f.lineno = 0
    f.col_offset = 0
    return f