from numerous.declarative import Connector, Module
from numerous.declarative.connector import get_value_for, set_value_from

import pytest

def test_get_connection():
    a = Module()
    b = Module()


    connector1 = Connector(b=get_value_for(a))
    connector2 = Connector(b=set_value_from(b))

    connector1.connect(connector2)

    assert connector1.connection[0] == connector2


