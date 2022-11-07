import pytest
import math
import shutil
import os.path
import pandas as pd
import numpy as np

from numerous.engine.system.external_mappings import ExternalMappingElement
from numerous.utils.data_loader import InMemoryDataLoader, CSVDataLoader
from numerous.utils.historian import InMemoryHistorian
from numerous.engine.system.external_mappings.interpolation_type import InterpolationType
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Subsystem, Item, LoggerLevel
from numerous.multiphysics import EquationBase, Equation

try:
    FEPS = np.finfo(1.0).eps
except AttributeError:
    FEPS = 2.220446049250313e-16


def analytical_solution(tmax, g=9.81, f=0.05, x0=1):
    t_hits = []
    summation = 0
    i = 0
    while True:
        summation += (2 * (1 - f) ** (i))
        t_hit = np.sqrt(2 * x0 / g) * (summation - 1)
        if t_hit > tmax:
            break
        t_hits.append(t_hit)
        i += 1

    t_hits = np.array(t_hits)
    return t_hits


class StaticDataTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(StaticDataTest, self).__init__(tag)

        # will map to variable with the same path in external dataframe/datasource
        self.add_parameter('T1', 0)
        self.add_parameter('T2', 0)
        self.add_parameter('T_i1', 0)
        self.add_parameter('T_i2', 0)
        mechanics = self.create_namespace('test_nm')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.T_i1 = scope.T1
        scope.T_i2 = scope.T2


class SubSubSystem(Subsystem):
    def __init__(self, tag='system', n=3, external_mappings=None, data_loader=None):
        super(SubSubSystem, self).__init__(tag=tag, external_mappings=external_mappings, data_loader=data_loader)
        o_s = []
        for i in range(n):
            o = StaticDataSystem('sub' + str(i))
            o_s.append(o)

        for i in reversed(range(n - 1)):
            o = o_s[i]
            o_sub = o_s[i + 1]
            o.register_item(o_sub)

        self.register_item(o_s[0])


class StaticDataSystem(Subsystem):
    def __init__(self, tag, n=1, external_mappings=None, data_loader=None):
        super().__init__(tag, external_mappings, data_loader)
        o_s = []
        for i in range(n):
            o = StaticDataTest('tm' + str(i))
            o_s.append(o)
        # Register the items to the subsystem to make it recognize them.
        self.register_items(o_s)


class OuterSystem(Subsystem):
    def __init__(self, tag, system, n=1, external_mappings=None, data_loader=None):
        super().__init__(tag, external_mappings, data_loader)
        o_s = []
        for i in range(n):
            o = StaticDataTest('tm' + str(i))
            o_s.append(o)
        o_s.append(system)
        # Register the items to the subsystem to make it recognize them.
        self.register_items(o_s)


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
    def __init__(self, tag="ball", g=9.81, f_loss=0.05, x0=1, v0=0):
        super().__init__(tag)

        t1 = self.create_namespace('t1')
        t1.add_equations([Bouncing(g=g, f_loss=f_loss, x=x0, v=v0)])

        def hitground_event_fun(t, states):
            return states['t1.x']

            # change direction of movement upon event detection and reduce velocity

        def hitground_event_callback_fun(t, variables):
            velocity = variables['t1.v']
            velocity = -velocity * (1 - variables['t1.f_loss'])
            variables['t1.v'] = velocity
            variables['t1.t_hit'] = t

        self.add_event("hitground_event", hitground_event_fun, hitground_event_callback_fun)


class StaticDataSystemWithBall(Subsystem):
    def __init__(self, tag, n=1, external_mappings=None, data_loader=None):
        super().__init__(tag, external_mappings, data_loader)
        o_s = []
        for i in range(n):
            o = StaticDataTest('tm' + str(i))
            o_s.append(o)
        o_s.append(Ball())
        # Register the items to the subsystem to make it recognize them.
        self.register_items(o_s)


def step_solver(sim, t0: float, tmax: float, dt: float):
    t_ = t0
    while abs(t_ - tmax) > 1e-6:
        t_new, t_ = sim.step_solve(t_, dt)

    sim.model.create_historian_df()
    df = sim.model.historian_df
    return df


def normal_solver(sim, t0: float, tmax: float, dt: float):
    sim.solve()

    df = sim.model.historian_df
    return df


def inmemorydataloader(df=None, **kwargs):
    return InMemoryDataLoader(df=df)


def csvdataloader(chunksize=1, **kwargs):
    return CSVDataLoader(chunksize=chunksize)


@pytest.fixture
def tmpdir():
    os.mkdir('tmp')
    yield 'tmp'
    shutil.rmtree('tmp')


@pytest.fixture
def external_data():
    def fn(t0=0, tmax=5, dt_data=0.5):
        data = {'time': np.arange(t0, tmax + dt_data, dt_data),
                'Dew Point Temperature {C}': np.arange(t0, tmax + dt_data, dt_data) + 1,
                }
        return data
    yield fn


@pytest.fixture
def simulation(tmpdir: tmpdir):
    def fn(data, dt_eval=0.1, chunksize=1, historian_max_size=None, t0=0, tmax=5, max_step=0.1, dataloader=csvdataloader,
           system=StaticDataSystemWithBall, mapped_name="Dew Point Temperature {C}",
           variable_name='system_external.tm0.test_nm.T1', use_llvm=True):
        external_mappings = []

        df = pd.DataFrame(data, columns=['time', 'Dew Point Temperature {C}'])
        index_to_timestep_mapping = 'time'
        index_to_timestep_mapping_start = 0

        dataframe_aliases = {
            variable_name: (mapped_name, InterpolationType.PIESEWISE),
        }
        path = os.path.join(tmpdir, "x.csv")
        df.to_csv(path)
        external_mappings.append(ExternalMappingElement
                                 (path, index_to_timestep_mapping, index_to_timestep_mapping_start, 1,
                                  dataframe_aliases))
        data_loader = dataloader(df=df, chunksize=chunksize)

        historian = InMemoryHistorian()
        historian.max_size = historian_max_size
        s = Simulation(
            Model(system
                  ('system_external', external_mappings=external_mappings, data_loader=data_loader),
                  historian=historian, use_llvm=use_llvm),
            t_start=t0, t_stop=tmax, num=len(np.arange(0, tmax, dt_eval)), max_step=max_step)
        return s
    yield fn


@pytest.mark.parametrize("system", [StaticDataSystem, StaticDataSystemWithBall])
@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_data_multiple(use_llvm, system, external_data):
    external_mappings = []
    external_mappings_outer = []

    t0 = 0
    tmax = 5
    dt_data = 0.1
    dt_eval = 0.1
    data = external_data(t0, tmax, dt_data)
    data.update({'Dry Bulb Temperature {C}': data['Dew Point Temperature {C}'] + 1})

    df = pd.DataFrame(data, columns=['time', 'Dew Point Temperature {C}', 'Dry Bulb Temperature {C}'])
    index_to_timestep_mapping = 'time'
    index_to_timestep_mapping_start = 0
    dataframe_aliases = {
        'system_outer.system_external.tm0.test_nm.T1': ("Dew Point Temperature {C}", InterpolationType.PIESEWISE),
        'system_outer.system_external.tm0.test_nm.T2': ('Dry Bulb Temperature {C}', InterpolationType.PIESEWISE),
    }
    dataframe_aliases_outer = {
        'system_outer.tm0.test_nm.T1': ("Dew Point Temperature {C}", InterpolationType.PIESEWISE),
        'system_outer.tm0.test_nm.T2': ('Dry Bulb Temperature {C}', InterpolationType.PIESEWISE),
    }
    external_mappings.append(ExternalMappingElement
                             ("inmemory", index_to_timestep_mapping, index_to_timestep_mapping_start, 1,
                              dataframe_aliases))
    external_mappings_outer.append((ExternalMappingElement
                                    ("inmemory", index_to_timestep_mapping, index_to_timestep_mapping_start, 1,
                                     dataframe_aliases_outer)))
    data_loader = InMemoryDataLoader(df)
    system_outer = OuterSystem('system_outer',
                               system('system_external', n=1, external_mappings=external_mappings,
                                                data_loader=data_loader),
                               external_mappings=external_mappings_outer,
                               data_loader=data_loader)
    s = Simulation(
        Model(system_outer, use_llvm=use_llvm),
        t_start=0, t_stop=tmax, num=len(np.arange(0, tmax, dt_eval)), max_step=0.1)
    s.solve()
    df = s.model.historian_df

    for ix in df.index:
        t = df['time'][ix]
        ix_data = math.floor(t / dt_data + FEPS * 100)
        v0 = data['Dew Point Temperature {C}'][ix_data]
        z0 = data['Dry Bulb Temperature {C}'][ix_data]
        v1 = df['system_outer.tm0.test_nm.T1'][ix]
        v2 = df['system_outer.tm0.test_nm.T_i1'][ix]
        v3 = df['system_outer.system_external.tm0.test_nm.T1'][ix]
        v4 = df['system_outer.system_external.tm0.test_nm.T_i1'][ix]
        z1 = df['system_outer.tm0.test_nm.T2'][ix]
        z2 = df['system_outer.tm0.test_nm.T_i2'][ix]
        z3 = df['system_outer.system_external.tm0.test_nm.T2'][ix]
        z4 = df['system_outer.system_external.tm0.test_nm.T_i2'][ix]

        assert v1 == pytest.approx(v0), f"expected {v0} but got {v1} at {t}"
        assert v2 == pytest.approx(v0), f"expected {v0} but got {v2} at {t}"
        assert v3 == pytest.approx(v0), f"expected {v0} but got {v3} at {t}"
        assert v4 == pytest.approx(v0), f"expected {v0} but got {v4} at {t}"
        assert v3 == pytest.approx(v4), f"expected {v3} but got {v4} at {t}"

        assert z1 == pytest.approx(z0), f"expected {z0} but got {z1} at {t}"
        assert z2 == pytest.approx(z0), f"expected {z0} but got {z2} at {t}"
        assert z3 == pytest.approx(z0), f"expected {z0} but got {z3} at {t}"
        assert z4 == pytest.approx(z0), f"expected {z0} but got {z4} at {t}"
        assert z3 == pytest.approx(z4), f"expected {z3} but got {z4} at {t}"


@pytest.mark.parametrize("use_llvm", [False, True])
@pytest.mark.parametrize("chunksize", [1, 10000])
@pytest.mark.parametrize("historian_max_size", [1, 10000])
@pytest.mark.parametrize("solver", [step_solver, normal_solver])
@pytest.mark.parametrize("dataloader", [inmemorydataloader, csvdataloader])
@pytest.mark.parametrize("system", [StaticDataSystem, StaticDataSystemWithBall])
# @pytest.mark.parametrize("use_llvm", [False])
# @pytest.mark.parametrize("chunksize", [1])
# @pytest.mark.parametrize("historian_max_size", [1])
# @pytest.mark.parametrize("solver", [step_solver])
# @pytest.mark.parametrize("dataloader", [csvdataloader])
# @pytest.mark.parametrize("system", [StaticDataSystemWithBall])
def test_external_data_chunks_and_historian_update(external_data: external_data, simulation: simulation,
                                                   solver, chunksize, historian_max_size, dataloader, system, use_llvm):

    t0 = 0
    tmax = 5
    dt = 0.1
    dt_data = 0.5
    data = external_data(t0, tmax, dt_data)
    s = simulation(data=data, chunksize=chunksize, historian_max_size=historian_max_size, t0=t0, max_step=dt,
                   dt_eval=dt, tmax=tmax,
                   dataloader=dataloader, system=system, use_llvm=use_llvm)
    df = solver(s, t0, tmax, dt)

    for ix in df.index:
        t = df['time'][ix]
        ix_data = math.floor(t / dt_data + FEPS * 100)
        v0 = data['Dew Point Temperature {C}'][ix_data]
        v = df['system_external.tm0.test_nm.T1'][ix]
        v1 = df['system_external.tm0.test_nm.T_i1'][ix]

        assert v == pytest.approx(v0), f"expected {v0} but got {v}"
        assert v1 == pytest.approx(v), f"expected {v} but got {v1}"

    t_hit_analytical = []
    if system == StaticDataSystemWithBall:
        t_hit_analytical = analytical_solution(tmax)
        t_hit_model = np.unique(df['system_external.ball.t1.t_hit'])[1:]
        assert len(t_hit_model) == len(t_hit_analytical), "not all events were detected"
        assert t_hit_model == pytest.approx(t_hit_analytical, 1e-3), "event detection inaccurate"

    assert len(df['system_external.tm0.test_nm.T1']) - len(t_hit_analytical) == len(np.arange(0, tmax + dt, dt)), \
        "output length not as expected"


def test_csvdataloader_reset(external_data: external_data, simulation: simulation):
    t0 = 0
    tmax = 5
    dt = 0.1
    dt_data = 0.5
    data = external_data(t0, tmax, dt_data)
    chunksize = 1
    historian_max_size = 10000
    s = simulation(data=data, chunksize=chunksize, historian_max_size=historian_max_size, t0=t0, tmax=tmax, max_step=dt,
                   dt_eval=dt, dataloader=csvdataloader, system=StaticDataSystemWithBall)
    s.solve()
    s.solve()


def test_wrong_mapping(external_data: external_data, simulation: simulation):
    t0 = 0
    tmax = 5
    dt = 0.1
    dt_data = 0.5
    data = external_data(t0, tmax, dt_data)
    chunksize = 10000
    historian_max_size = 10000
    with pytest.raises(ValueError, match=r"No variable named*"):
        simulation(data=data, chunksize=chunksize, historian_max_size=historian_max_size, t0=t0, tmax=tmax,
                       max_step=dt, dt_eval=dt, dataloader=csvdataloader, system=StaticDataSystem,
                       variable_name='does.not.exist')

    simulation(data=data, chunksize=chunksize, historian_max_size=historian_max_size, t0=t0, tmax=tmax,
                       max_step=dt, dt_eval=dt, dataloader=csvdataloader, system=SubSubSystem,
                       variable_name='system_external.sub0.sub1.sub2.tm0.test_nm.T1')


class SimpleInt(Subsystem, EquationBase):
    def __init__(self, tag='integrate', external_mappings=None, data_loader=None):
        super().__init__(tag, external_mappings, data_loader)
        self.t1 = self.create_namespace('t1')
        self.add_state('x', 0, logger_level=LoggerLevel.INFO)
        self.add_parameter('to_map', 0)
        self.add_parameter('not_to_map', 100)
        self.add_parameter('to_map_think_not', 200)

        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        scope.x_dot = 1


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_data_model_check_not_mapped(use_llvm):
    external_mappings = []

    data = {'time': np.arange(101),
            'to_map': np.arange(101),
            }

    df = pd.DataFrame(data)
    index_to_timestep_mapping = 'time'
    index_to_timestep_mapping_start = 0
    dataframe_aliases = {
        'system_external.t1.to_map': ("to_map", InterpolationType.PIESEWISE),
    }
    external_mappings.append(ExternalMappingElement
                             ("inmemory", index_to_timestep_mapping, index_to_timestep_mapping_start, 1,
                              dataframe_aliases))
    data_loader = InMemoryDataLoader(df)

    m = Model(SimpleInt('system_external'), use_llvm=use_llvm)

    s = Simulation(
        m,
        t_start=0, t_stop=100.0, num=100, num_inner=1, max_step=1)

    m.set_external_mappings(external_mappings, data_loader=data_loader)

    s.solve()
    assert pytest.approx(np.array(s.model.historian_df['system_external.t1.to_map'])) == np.arange(101), \
        'This variable should have the values as being mapped'
    assert pytest.approx(np.array(s.model.historian_df['system_external.t1.not_to_map'])) == 100, \
        'This should not be mapped and thus not changed'
    assert pytest.approx(np.array(s.model.historian_df['system_external.t1.to_map_think_not'])) == 200, \
        'This should not be mapped and thus not changed'


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_data_system_check_not_mapped(use_llvm):
    external_mappings = []

    data = {'time': np.arange(101),
            'to_map': np.arange(101),
            }

    df = pd.DataFrame(data)
    index_to_timestep_mapping = 'time'
    index_to_timestep_mapping_start = 0
    dataframe_aliases = {
        'system_external.t1.to_map': ("to_map", InterpolationType.PIESEWISE),
    }
    external_mappings.append(ExternalMappingElement
                             ("inmemory", index_to_timestep_mapping, index_to_timestep_mapping_start, 1,
                              dataframe_aliases))
    data_loader = InMemoryDataLoader(df)

    m = Model(SimpleInt('system_external', data_loader=data_loader, external_mappings=external_mappings),
              use_llvm=use_llvm)

    s = Simulation(
        m,
        t_start=0, t_stop=100.0, num=100, num_inner=1, max_step=1)

    s.solve()
    assert pytest.approx(np.array(s.model.historian_df['system_external.t1.to_map'])) == np.arange(101), \
        'This variable should have the values as being mapped'
    assert pytest.approx(np.array(s.model.historian_df['system_external.t1.not_to_map'])) == 100, \
        'This should not be mapped and thus not changed'
    assert pytest.approx(np.array(s.model.historian_df['system_external.t1.to_map_think_not'])) == 200, \
        'This should REALLY not be mapped and thus not changed'


class TwoSystemInside(Subsystem):
    def __init__(self, tag, system, n=1, external_mappings=None, data_loader=None):
        super().__init__(tag, external_mappings, data_loader)
        o_s = []
        o_s.append(StaticDataSystem(tag='tag', n=1))

        o_s.append(system)

        # Register the items to the subsystem to make it recognize them.
        self.register_items(o_s)


@pytest.mark.parametrize("use_llvm", [True, False])
def test_get_external_system_fix(use_llvm, external_data: external_data):
    t0 = 0
    tmax = 5
    dt_data = 0.1
    external_mappings = []
    data = external_data(t0, tmax, dt_data)
    data.update({'Dry Bulb Temperature {C}': data['Dew Point Temperature {C}'] + 1})

    df = pd.DataFrame(data, columns=['time', 'Dew Point Temperature {C}', 'Dry Bulb Temperature {C}'])
    index_to_timestep_mapping = 'time'
    index_to_timestep_mapping_start = 0
    dataframe_aliases = {
        'system_outer.system_external.tm0.test_nm.T1': ("Dew Point Temperature {C}", InterpolationType.PIESEWISE),
        'system_outer.system_external.tm0.test_nm.T2': ('Dry Bulb Temperature {C}', InterpolationType.PIESEWISE),
    }

    external_mappings.append(ExternalMappingElement
                             ("inmemory", index_to_timestep_mapping, index_to_timestep_mapping_start, 1,
                              dataframe_aliases))

    data_loader = InMemoryDataLoader(df)
    system_outer = TwoSystemInside('system_outer',
                                   StaticDataSystem('system_external', n=1, external_mappings=external_mappings,
                                                    data_loader=data_loader),
                                    data_loader=data_loader)
    dt_eval = 0.1
    s = Simulation(
        Model(system_outer, use_llvm=use_llvm),
        t_start=0, t_stop=tmax, num=len(np.arange(0, tmax, dt_eval)), max_step=0.1)
    s.solve()
