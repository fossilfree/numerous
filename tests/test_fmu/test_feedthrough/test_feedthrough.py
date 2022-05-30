import os

import numpy as np
import pandas as pd
import pytest

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Subsystem
from numerous.engine.system.fmu_subsystem import FMU_Subsystem


class S3(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)

        fmu_filename = os.path.join(os.path.dirname(__file__), 'Feedthrough.fmu')
        fmu_in = os.path.join(os.path.dirname(__file__), 'Feedthrough_in.csv')
        fmu_subsystem = FMU_Subsystem(fmu_filename, "Feedthrough", fmu_in=fmu_in, debug_output=False)
        self.register_items([fmu_subsystem])


@pytest.mark.parametrize("use_llvm", [True, False])
def test_feedthrough(use_llvm):
    subsystem = S3('q1')

    m1 = Model(subsystem, use_llvm=use_llvm)
    s = Simulation(
        m1, t_start=0, t_stop=2, num=10, num_inner=1000, max_event_steps=.1)
    s.solve()
    df = s.model.historian_df
    assert np.allclose(df['q1.Feedthrough.t1.real_discrete_in'], np.array([0.0] * 5 + [1.0] * 6), rtol=1e-4)
    assert np.allclose(df['q1.Feedthrough.t1.real_continuous_in'], np.array([0.0] * 3 + [0.2] + [0.6] + [1.0] * 6), rtol=1e-4)
