from numerous.declarative import Connector, ItemsSpec, Module, EquationSpec, create_connections
from numerous.declarative.connector import get_value_for, set_value_from

import pytest

def test_get_connection():
    a = Module()
    b = Module()


    connector1 = Connector(a=get_value_for(a))
    connector2 = Connector(b=set_value_from(b))

    with create_connections() as connections:
        connector1.connect(connector2)

    assert connector1.get_connection() == connector2


