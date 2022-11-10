from numerous.declarative.specification import ItemsSpec, Module, ScopeSpec, EquationSpec
from numerous.declarative.mappings import create_mappings
from numerous.declarative.exceptions import MappingOutsideMappingContextError
from numerous.declarative.utils import print_map
from numerous.declarative.variables import Parameter

from numerous.engine import model, simulation
from .test_mappings import TestCompositeNodeConnector

from numerous.declarative.debug_utils import print_all_variables
import pytest

def test_print_all_var():
    F=1
    composite = TestCompositeNodeConnector("composite", F=F)
    composite.finalize()

    m = model.Model(composite, use_llvm=False)

    # Define simulation
    s = simulation.Simulation(
        m,
        t_start=0, t_stop=10, num=10, num_inner=1, max_step=10
    )
    # Solve
    s.solve()

    print_all_variables(composite, s)
