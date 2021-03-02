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

def recurse_Attribute(attr, sep='.'):
        if hasattr(attr, 'id'):
            return attr.id
        elif isinstance(attr.value, ast.Name):
            return attr.value.id + sep + attr.attr
        elif isinstance(attr.value, ast.Attribute):
            return recurse_Attribute(attr.value) + sep + attr.attr

class dot_dict:
        def __init__(self, **d):
            for k, v in d.items():
                setattr(self, k, v)


def wrap_function(name, body, args, decorators):
    f = ast.FunctionDef(name)
    f.body = body
    f.decorator_list = decorators
    f.args = args
    return f