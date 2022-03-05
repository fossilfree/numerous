import numpy as np
import pytest
from numerous.engine.system import Item
from numerous.engine.model import Model
from numerous.engine.system import Subsystem
from numerous.engine.simulation import Simulation
from numerous.multiphysics import EquationBase, Equation
from pytest import approx

tmax = 5
num = 1000


def analytical_solution(N_hits, g=9.81, f=0.05, x0=1):
    t_hits = []
    summation = 0
    for i in range(N_hits):
        summation += (2 * (1 - f) ** (i))
        t_hit = np.sqrt(2 * x0 / g) * (summation - 1)
        t_hits.append(t_hit)
    t_hits = np.array(t_hits)
    return t_hits


t_hits = analytical_solution(N_hits=10)


class Bouncing(EquationBase):
    def __init__(self, g=9.81, f_loss=0.05, x=1, v=0):
        super().__init__(tag='bouncing_eq')
        self.add_constant('g', g)
        self.add_constant('f_loss', f_loss)
        self.add_parameter('t_hit', 0)
        self.add_state('x', x)
        self.add_state('v', v)

    @Equation()
    def eval(self, scope):
        scope.x_dot = scope.v  # Position
        scope.v_dot = -scope.g  # Velocity


class Ball(Item):
    def __init__(self, tag="ball", g=9.81, f_loss=5, x0=1, v0=0):
        super().__init__(tag)

        t1 = self.create_namespace('t1')
        t1.add_equations([Bouncing(g=g, f_loss=f_loss, x=x0, v=v0)])

        # returns position to find zero crossing using root finding algorithm of scipy solver
        def hitground_event_fun(t, states):
            return states['t1.x']

        # change direction of movement upon event detection and reduce velocity
        def hitground_event_callback_fun(t, variables):
            velocity = variables['t1.v']
            velocity = -velocity * (1 - variables['t1.f_loss'])
            variables['t1.v'] = velocity
            variables['t1.t_hit'] = t

        self.add_event("hitground_event", hitground_event_fun, hitground_event_callback_fun)


class Ball2(Item):
    def __init__(self, tag="ball", g=9.81, f_loss=5, x0=1, v0=0):
        super().__init__(tag)

        t1 = self.create_namespace('t1')
        t1.add_equations([Bouncing(g=g, f_loss=f_loss, x=x0, v=v0)])

        def timestamp_callback(t, variables):
            print(t)

        self.add_timestamp_event("timestamp_event", timestamp_callback, timestamps=[0.11, 0.33])


def ms1(simple_item):
    class S1(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            self.register_items([simple_item])

    return S1('S1')


@pytest.mark.parametrize("use_llvm", [True, False])
def test_bouncing_ball(use_llvm):
    model_system_2 = ms1(Ball(tag="ball", g=9.81, f_loss=0.05))
    m1 = Model(model_system_2, use_llvm=use_llvm)

    sim = Simulation(m1, t_start=0, t_stop=tmax, num=num)

    sim.solve()
    asign = np.sign(np.array(m1.historian_df['S1.ball.t1.v']))
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    args = np.argwhere(signchange > 0)[2:].flatten()
    assert approx(m1.historian_df['time'][args[0::2][:5]], rel=0.01) == t_hits[:5]


@pytest.mark.parametrize("use_llvm", [True, False])
def test_timetamp_item_event(use_llvm, capsys):
    model_system_2 = ms1(Ball2(tag="ball", g=9.81, f_loss=0.05))
    m1 = Model(model_system_2, use_llvm=use_llvm)

    sim = Simulation(m1, t_start=0, t_stop=tmax, num=num)

    sim.solve()
    captured = capsys.readouterr()

    assert captured.out == "0.11\n0.33\n"
