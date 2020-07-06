from enum import IntEnum, unique

@unique
class NodeTypes(IntEnum):
    OP=0
    ASSIGN = 1
    VAR=2
    DERIV=3
    STATE=4