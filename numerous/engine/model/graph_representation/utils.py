import ast
from enum import unique, IntEnum


class Vardef:
    def __init__(self):
        self.vars_inds_map = []

    def var_def(self, var, read=True):
        if not var in self.vars_inds_map:
            self.vars_inds_map.append(var)
        ix = self.vars_inds_map.index(var)

        return ast.Subscript(slice=ast.Index(value=ast.Num(n=ix)), value=ast.Name(id='l'))


@unique
class NodeTypes(IntEnum):
    OP = 0
    ASSIGN = 1
    VAR = 2
    EQUATION = 3
    SUM = 4
    TMP = 5


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