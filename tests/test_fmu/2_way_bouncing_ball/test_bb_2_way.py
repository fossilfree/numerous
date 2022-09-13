import os

import numpy as np
import pytest
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Subsystem
from numerous.engine.system.fmu_subsystem import FMU_Subsystem
from pytest import approx


class S3(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)
        fmu_filename = os.path.join(os.path.dirname(__file__), 'BouncingBall_2_way.fmu')
        fmu_subsystem = FMU_Subsystem(fmu_filename, "BouncingBall_2_way", debug_output=True)
        self.register_items([fmu_subsystem])



@pytest.mark.parametrize("use_llvm", [True, False])
def test_bounsing_ball_2_way_pos_hits(use_llvm):
    subsystem1 = S3('q1')
    m1 = Model(subsystem1, use_llvm=use_llvm)
    s = Simulation(m1, t_start=0, t_stop=1, num=10, num_inner=1, max_step=.1)
    s.solve()
    pos = np.array(m1.historian_df["q1.BouncingBall_2_way.t1.h"])
    vel = m1.historian_df["q1.BouncingBall_2_way.t1.v"]
    asign = np.sign(np.array(vel))
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    args = np.argwhere(signchange > 0)[2:].flatten()
    assert all(np.isclose(pos[args], np.array([0, 1, 0]), rtol=1e-03, atol=1e-01))



@pytest.mark.parametrize("use_llvm", [True, False])
def test_bounsing_ball_2_way_t_hits(use_llvm):
    subsystem1 = S3('q1')
    m1 = Model(subsystem1, use_llvm=use_llvm)
    s = Simulation(m1, t_start=0, t_stop=1, num=10, num_inner=1, max_step=.1)
    s.solve()
    vel = m1.historian_df["q1.BouncingBall_2_way.t1.v"]
    asign = np.sign(np.array(vel))
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    args = np.argwhere(signchange > 0)[2:].flatten()
    t_hits = [0.44928425, 0.69429896, 0.91407356]
    assert approx(np.array(m1.historian_df["time"][args]), rel=0.01) == t_hits
