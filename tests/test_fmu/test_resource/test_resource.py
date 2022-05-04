import numpy as np
import pandas as pd
import pytest

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation, SolverType
from numerous.engine.simulation.solvers import solver_types
from numerous.engine.system import Subsystem
from numerous.engine.system.fmu_subsystem import FMU_Subsystem


class S3(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)

        fmu_filename = 'Resource.fmu'
        fmu_subsystem = FMU_Subsystem(fmu_filename, "Resource", debug_output=True)
        self.register_items([fmu_subsystem])


@pytest.mark.xfail
@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_resource(solver, use_llvm):
    subsystem1 = S3('q1')
    m1 = Model(subsystem1, use_llvm=use_llvm)
    s = Simulation(
        m1, t_start=0, t_stop=1, num=6, num_inner=1000, max_step=.1, solver_type=solver)
    s.solve()
    h = np.array(m1.historian_df["q1.Resource.t1.y"])
    df = pd.read_csv("Resource_ref.csv")
    assert np.allclose(h, df['h'], rtol=1e-1)
