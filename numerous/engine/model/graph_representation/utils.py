import ast
from enum import IntEnum, unique


class TemporaryKeyGenerator:
    def __init__(self):
        self.tmp_ = 0

    def generate(self):
        self.tmp_ += 1
        return f'tmp{self.tmp_}'


class Vardef:
    def __init__(self):
        self.vars_inds_map = []

    def var_def(self, var, read=True):
        if not var in self.vars_inds_map:
            self.vars_inds_map.append(var)
        ix = self.vars_inds_map.index(var)

        return ast.Subscript(slice=ast.Index(value=ast.Constant(value=ix)), value=ast.Name(id='l'))


@unique
class EdgeType(IntEnum):
    TARGET = 0
    ARGUMENT = 1
    MAPPING = 2
    UNDEFINED = 3
    OPERAND = 4
    TMP = 5
    VALUE = 6
    LEFT = 7
    RIGHT = 8
    BODY = 9
    ORELSE = 10
    TEST = 11
    COMP = 12
    DEP = 13
    SLICE = 14
    SUBSCRIPT_VALUE = 15
    ELEMENT = 16


def str_to_edgetype(a):
    if a == "left":
        return EdgeType.LEFT
    if a == "right":
        return EdgeType.RIGHT
    if a == 'body':
        return EdgeType.BODY
    if a == 'orelse':
        return EdgeType.ORELSE
    if a == 'test':
        return EdgeType.TEST
    if a == 'subscript_value':
        return EdgeType.SUBSCRIPT_VALUE
    if a == 'slice':
        return EdgeType.SLICE
