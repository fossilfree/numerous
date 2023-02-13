import pytest
import numpy as np
from numba.core.typeinfer import TypingError
from pytest import approx
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Subsystem
from numerous.utils.logger_levels import LoggerLevel
from numerous.multiphysics import EquationBase, Equation

def analytical_solution(N_hits, g=9.81, f=0.05, x0=1):
    t_hits = []
    summation = 0
    for i in range(N_hits):
        summation += (2 * (1 - f) ** (i))
        t_hit = np.sqrt(2 * x0 / g) * (summation - 1)
        t_hits.append(t_hit)
    t_hits = np.array(t_hits)
    return t_hits


def event_condition(path):
    def bounce_event_condition(t, variables):
        position = path + 't1.x'
        return variables[position]

    return bounce_event_condition

def event_action(path, external: bool, print_string: bool):

    def bounce_event_action_internal(t, variables):
        velocity = path + "t1.v"
        f_loss = path + "t1.f_loss"
        t_hit = path + 't1.t_hit'
        if print_string:
            print("this works as an internal event event function")

        variables[velocity] = - variables[velocity] * (1 - variables[f_loss])
        variables[t_hit] = t

    def bounce_event_action_external(t, variables):
        velocity = path + "t1.v"
        f_loss = path + "t1.f_loss"
        t_hit = path + 't1.t_hit'
        class A:
            def __init__(self):
                if print_string:
                    print("this will not work as an internal event function")
                variables[velocity] = - variables[velocity] * (1 - variables[f_loss])
                variables[t_hit] = t
        A()

    return bounce_event_action_internal if not external else bounce_event_action_external

def event_action_model_with_wrong_path(t, variables):
    velocity = variables['system.t1.']
    velocity = -velocity * (1 - variables['system.t1.f_loss'])
    variables['system.t1.v'] = velocity
    variables['system.t1.t_hit'] = t

class BouncingBall(Subsystem, EquationBase):
    def __init__(self, tag='system', g=9.81, f_loss=0.05, x=1, v=0):
        super().__init__(tag=tag)
        self.t1 = self.create_namespace('t1')
        self.add_constant('g', g)
        self.add_constant('f_loss', f_loss)
        self.add_parameter('t_hit', 0)
        self.add_state('x', x)
        self.add_state('v', v)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x_dot = scope.v  # Position
        scope.v_dot = -scope.g  # Velocity


class ExponentialDecay(Subsystem, EquationBase):
    def __init__(self, tag='exp', alpha=0.1):
        super(ExponentialDecay, self).__init__(tag)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 1, logger_level=LoggerLevel.INFO)
        self.add_constant('alpha', alpha)
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x_dot = -scope.x * scope.alpha

@pytest.mark.parametrize("use_llvm", [True, False])
def test_timestamp_event_item(use_llvm, capsys):
    sys = ExponentialDecay(tag='system')
    def time_callback(t, variables):
        print(t)

    timestamps = [float((i+1) * 3600) for i in range(24)]

    sys.add_timestamp_event('test', time_callback, timestamps=timestamps)
    model = Model(sys, use_llvm=use_llvm)
    #num not aligned with timestamps output
    sim = Simulation(model=model, t_start=0, t_stop=24*3600, num=33)
    sim.solve()

    captured = capsys.readouterr()

    assert captured.out == "\n".join([str(t) for t in timestamps])+"\n"

@pytest.mark.parametrize("use_llvm", [True, False])
def test_timestamp_event_model(use_llvm, capsys):
    sys = ExponentialDecay(tag='system')
    def time_callback(t, variables):
        print(t)

    timestamps = [float((i+1) * 3600) for i in range(24)]

    model = Model(sys, use_llvm=use_llvm)
    model.add_timestamp_event('test', time_callback, timestamps=timestamps)
    #num not aligned with timestamps output
    sim = Simulation(model=model, t_start=0, t_stop=24*3600, num=33)
    sim.solve()

    captured = capsys.readouterr()

    assert captured.out == "\n".join([str(t) for t in timestamps])+"\n"

@pytest.mark.parametrize("use_llvm", [False, True])
def test_multiple_timestamp_events_item(use_llvm, capsys):
    sys = ExponentialDecay(tag='system')
    def time_callback1(t, variables):
        print("callback1:", t)

    def time_callback2(t, variables):
        print("callback2:", t)

    timestamps1 = [float((i+1) * 3600) for i in range(24)]
    timestamps2 = [float((i+1) * 7200) for i in range(12)]

    sys.add_timestamp_event('callback1', time_callback1, timestamps=timestamps1)
    sys.add_timestamp_event('callback2', time_callback2, timestamps=timestamps2)
    model = Model(sys, use_llvm=use_llvm)
    #num not aligned with timestamps output
    sim = Simulation(model=model, t_start=0, t_stop=24*3600, num=33)
    sim.solve()

    captured = capsys.readouterr()
    expected = []
    for t in timestamps1:
        s = f"callback1: {t}\n"
        if t in timestamps2:
            s += f"callback2: {t}\n"

        expected.append(s)

    assert captured.out == "".join(expected)

@pytest.mark.parametrize("use_llvm", [False, True])
def test_multiple_timestamp_events_model(use_llvm, capsys):
    sys = ExponentialDecay(tag='system')
    def time_callback1(t, variables):
        print("callback1:", t)

    def time_callback2(t, variables):
        print("callback2:", t)

    timestamps1 = [float((i+1) * 3600) for i in range(24)]
    timestamps2 = [float((i+1) * 7200) for i in range(12)]

    model = Model(sys, use_llvm=use_llvm)
    model.add_timestamp_event('callback1', time_callback1, timestamps=timestamps1)
    model.add_timestamp_event('callback2', time_callback2, timestamps=timestamps2)
    #num not aligned with timestamps output
    sim = Simulation(model=model, t_start=0, t_stop=24*3600, num=33)
    sim.solve()

    captured = capsys.readouterr()
    expected = []
    for t in timestamps1:
        s = f"callback1: {t}\n"
        if t in timestamps2:
            s += f"callback2: {t}\n"

        expected.append(s)

    assert captured.out == "".join(expected)

@pytest.mark.parametrize("use_llvm", [False, True])
def test_multiple_timestamp_events_mixed(use_llvm, capsys):
    sys = ExponentialDecay(tag='system')
    def time_callback1(t, variables):
        print("callback1:", t)

    def time_callback2(t, variables):
        print("callback2:", t)

    timestamps1 = [float((i+1) * 3600) for i in range(24)]
    timestamps2 = [float((i+1) * 7200) for i in range(12)]
    sys.add_timestamp_event('callback1', time_callback1, timestamps=timestamps1)

    model = Model(sys, use_llvm=use_llvm)
    model.add_timestamp_event('callback2', time_callback2, timestamps=timestamps2)
    #num not aligned with timestamps output
    sim = Simulation(model=model, t_start=0, t_stop=24*3600, num=33)
    sim.solve()

    captured = capsys.readouterr()
    expected = []
    for t in timestamps1:
        s = f"callback1: {t}\n"
        if t in timestamps2:
            s += f"callback2: {t}\n"

        expected.append(s)

    assert captured.out == "".join(expected)

@pytest.mark.parametrize("use_llvm", [False, True])
def test_timestamp_periodicity(use_llvm, capsys):
    """
    Tests periodic time events
    """
    sys = ExponentialDecay(tag='system')
    def time_callback(t, variables):
        print(t)
    num = 48
    timestamps = [float((i) * 1800) for i in range(num+1)]

    # Use an input specifying the periodicity of the callbacks instead of a list of timestamps.
    sys.add_timestamp_event('test', time_callback, periodicity=1800.0)
    model = Model(sys, use_llvm=use_llvm)
    sim = Simulation(model=model, t_start=0, t_stop=24*3600, num=num)
    sim.solve()

    captured = capsys.readouterr()

    assert captured.out == "\n".join([str(t) for t in timestamps])+"\n"

@pytest.mark.parametrize("use_llvm", [False, True])
def test_periodicity_and_timestamp_lists(use_llvm, capsys):
    """
    Test both types of time events
    """
    sys = ExponentialDecay(tag='system')

    def time_callback1(t, variables):
        print("callback1:", t)

    def time_callback2(t, variables):
        print("callback2:", t)

    num = 48
    timestamps = [float((i) * 1800) for i in range(num+1)]

    # Use an input specifying the periodicity of the callbacks instead of a list of timestamps.
    sys.add_timestamp_event("periodic", time_callback1, periodicity=1800.0)
    sys.add_timestamp_event("list", time_callback2, timestamps=timestamps)
    model = Model(sys, use_llvm=use_llvm)
    sim = Simulation(model=model, t_start=0, t_stop=24*3600, num=num)
    sim.solve()

    captured = capsys.readouterr()

    expected = []
    for t in timestamps:
        s = f"callback1: {t}\n"
        if t in timestamps:
            s += f"callback2: {t}\n"

        expected.append(s)

    assert captured.out == "".join(expected)

@pytest.mark.parametrize("use_llvm", [False, True])
def test_item_state_events_accuracy(use_llvm):
    system = BouncingBall('S1')

    system.add_event('bounce', event_condition(""), event_action("", external=False, print_string=False), direction=-1)

    m1 = Model(system, use_llvm=use_llvm)
    atol = 1e-9
    sim = Simulation(m1, t_start=0, t_stop=5, num=10, atol=atol)

    sim.solve()
    model_hits = np.unique(m1.historian_df['S1.t1.t_hit'])[1:]
    num_hits = len(model_hits)
    t_hits = analytical_solution(num_hits)

    assert approx(model_hits, abs=1e-3) == t_hits

@pytest.mark.parametrize("use_llvm", [False, True])
def test_model_state_events_accuracy(use_llvm):
    system = BouncingBall('S1')

    m1 = Model(system, use_llvm=use_llvm)
    m1.add_event('bounce', event_condition("S1."), event_action("S1.", external=False, print_string=False),
                 direction=-1)
    atol = 1e-9
    sim = Simulation(m1, t_start=0, t_stop=5, num=10, atol=atol)

    sim.solve()
    model_hits = np.unique(m1.historian_df['S1.t1.t_hit'])[1:]
    num_hits = len(model_hits)
    t_hits = analytical_solution(num_hits)

    assert approx(model_hits, abs=1e-3) == t_hits

@pytest.mark.parametrize('is_external', [False, True])
@pytest.mark.parametrize('use_llvm', [False, True])
def test_item_external_state_events(is_external: bool, use_llvm: bool, capsys):
    system = BouncingBall(tag='system')
    system.add_event('bounce',
                     event_condition(""),
                     event_action("", external=is_external, print_string=True),
                     direction=-1,
                     is_external=is_external)

    model = Model(system=system, use_llvm=use_llvm)

    simulation = Simulation(model, t_start=0, t_stop=10, num=100)

    simulation.solve()

    captured = capsys.readouterr()

    search_string = "this works as an internal event event function" if not is_external else \
        "this will not work as an internal event function"

    assert search_string in captured.out

@pytest.mark.parametrize('is_external', [False, True])
@pytest.mark.parametrize('use_llvm', [False, True])
def test_model_external_state_events(is_external: bool, use_llvm: bool, capsys):
    system = BouncingBall(tag='system')

    model = Model(system=system, use_llvm=use_llvm)

    model.add_event('bounce',
                    event_condition("system."),
                    event_action("system.", external=is_external, print_string=True),
                    direction=-1,
                    is_external=is_external)

    simulation = Simulation(model, t_start=0, t_stop=10, num=100)

    simulation.solve()

    captured = capsys.readouterr()

    search_string = "this works as an internal event event function" if not is_external else \
        "this will not work as an internal event function"

    assert search_string in captured.out

@pytest.mark.parametrize("use_llvm", [True, False])
def test_item_timestamp_event_with_state_event(use_llvm, capsys):
    def timestamp_callback(t, variables):
        print(t)

    system = BouncingBall()
    system.add_event('bounce', event_condition(""), event_action("", external=False, print_string=False), direction=-1)
    m1 = Model(system, use_llvm=use_llvm)

    m1.add_timestamp_event("timestamp_event", timestamp_callback, timestamps=[0.11, 0.33])

    sim = Simulation(m1, t_start=0, t_stop=5, num=10, atol=1e-9)

    sim.solve()
    num_hits = len(np.unique(m1.historian_df['system.t1.t_hit'])[1:])

    captured = capsys.readouterr()

    assert captured.out == "0.11\n0.33\n"
    assert max(analytical_solution(num_hits)) < 5


@pytest.mark.parametrize("use_llvm", [True, False])
def test_model_timestamp_event_with_state_event(use_llvm, capsys):
    def timestamp_callback(t, variables):
        print(t)

    system = BouncingBall()
    m1 = Model(system, use_llvm=use_llvm)
    m1.add_event('bounce',
                 event_condition("system."),
                 event_action("system.", external=False, print_string=False),
                 direction=-1)

    m1.add_timestamp_event("timestamp_event", timestamp_callback, timestamps=[0.11, 0.33])

    sim = Simulation(m1, t_start=0, t_stop=5, num=10, atol=1e-9)

    sim.solve()
    num_hits = len(np.unique(m1.historian_df['system.t1.t_hit'])[1:])

    captured = capsys.readouterr()

    assert captured.out == "0.11\n0.33\n"
    assert max(analytical_solution(num_hits)) < 5

@pytest.mark.parametrize("use_llvm", [False])
def test_item_variable_error(use_llvm):
    with pytest.raises(KeyError):
        system = BouncingBall()
        system.add_event("hitground_event",
                         event_condition(""),
                         event_action("system.", external=False, print_string=False),
                         direction=-1)

        m1 = Model(system, use_llvm=use_llvm)

        sim = Simulation(m1, t_start=0, t_stop=5, num=100)

        sim.solve()

@pytest.mark.parametrize("use_llvm", [False])
def test_model_variable_error(use_llvm):
    with pytest.raises(KeyError):
        system = BouncingBall()

        m1 = Model(system, use_llvm=use_llvm)
        m1.add_event("hitground_event",
                     event_condition("system."),
                     event_action("", external=False, print_string=False),
                     direction=-1)

        sim = Simulation(m1, t_start=0, t_stop=5, num=100)

        sim.solve()

