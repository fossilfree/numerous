import numpy as np
import pandas as pd
import pytest

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.simulation.solvers import solver_types
from numerous.engine.system import Subsystem
from numerous.engine.system.fmu_subsystem import FMU_Subsystem


class S3(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)

        fmu_filename = 'Dahlquist.fmu'
        fmu_subsystem = FMU_Subsystem(fmu_filename, "Dahlquist", debug_output=True)
        self.register_items([fmu_subsystem])


@pytest.mark.xfail
@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_dahlquist(solver, use_llvm):
    subsystem1 = S3('q1')
    m1 = Model(subsystem1, use_llvm=use_llvm)
    s = Simulation(
        m1, t_start=0, t_stop=10, num=50, num_inner=1000, max_step=.1, solver_type=solver)
    s.solve()
    x = np.array(m1.historian_df["q1.Dahlquist.t1.x"])
    df = pd.read_csv("Dahlquist_ref.csv")
    assert np.allclose(x, df['x'], rtol=1e-1)
