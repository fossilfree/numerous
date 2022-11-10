from numerous.declarative.diagram import generate_diagram
from .test_mappings import TestModuleWithItems
import pytest

def test_diagram():

    generate_diagram(TestModuleWithItems("test"), view=False, sub_levels=10, include_variables=True)