from numerous.declarative.debug_utils import print_all_variables
from numerous.engine import model, simulation
from .test_mappings import TestCompositeNodeConnector


def test_print_all_var():
    F = 1
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

    print_all_variables(composite, s.model.historian_df)
