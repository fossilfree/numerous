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

        fmu_filename = 'VanDerPol.fmu'
        fmu_subsystem = FMU_Subsystem(fmu_filename, "VanDerPol", debug_output=True)
        self.register_items([fmu_subsystem])

@pytest.mark.xfail
@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_vanderpol(solver, use_llvm):
    subsystem1 = S3('q1')
    m1 = Model(subsystem1, use_llvm=use_llvm)
    s = Simulation(
        m1, t_start=0, t_stop=20, num=200, num_inner=1000, max_step=.1, solver_type=solver)
    s.solve()
    x0 = np.array(m1.historian_df["q1.VanDerPol.t1.x0"])
    x1 = np.array(m1.historian_df["q1.VanDerPol.t1.x1"])
    df = pd.read_csv("VanDerPol_ref.csv")
    assert np.allclose(x0, df['x0'], rtol=1e-1)
    assert np.allclose(x1, df['x1'], rtol=1e-1)
