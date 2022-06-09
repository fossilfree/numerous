import os.path

import pytest
import pandas as pd
import numpy as np
from numerous.engine.system.external_mappings import ExternalMappingElement

from numerous.utils.data_loader import InMemoryDataLoader, LocalDataLoader
from pytest import approx

from numerous.engine.system.external_mappings.interpolation_type import InterpolationType
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation

from numerous.engine.system import Subsystem, ConnectorItem, Item, ConnectorTwoWay, LoggerLevel
from numerous.multiphysics import EquationBase, Equation


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


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_data(use_llvm):
    external_mappings = []

    data = {'time': np.arange(100),
            'Dew Point Temperature {C}': np.arange(100) + 1,
            'Dry Bulb Temperature {C}': np.arange(100) + 2,
            }

    df = pd.DataFrame(data, columns=['time', 'Dew Point Temperature {C}', 'Dry Bulb Temperature {C}'])
    index_to_timestep_mapping = 'time'
    index_to_timestep_mapping_start = 0
    dataframe_aliases = {
        'system_external.tm0.test_nm.T1': ("Dew Point Temperature {C}", InterpolationType.PIESEWISE),
        'system_external.tm0.test_nm.T2': ('Dry Bulb Temperature {C}', InterpolationType.PIESEWISE)
    }
    external_mappings.append(ExternalMappingElement
                             ("inmemory", index_to_timestep_mapping, index_to_timestep_mapping_start, 1,
                              dataframe_aliases))
    data_loader = InMemoryDataLoader(df)
    Model(
        StaticDataSystemWithBall('system_external', n=1, external_mappings=external_mappings, data_loader=data_loader),
        use_llvm=use_llvm)

    s = Simulation(
        Model(StaticDataSystem('system_external', n=1, external_mappings=external_mappings, data_loader=data_loader),
              use_llvm=use_llvm),
        t_start=0, t_stop=100.0, num=100, num_inner=100, max_step=.1)

    s.solve()
    assert approx(np.array(s.model.historian_df['system_external.tm0.test_nm.T_i1'])[1:]) == np.arange(101)[1:]
    assert approx(np.array(s.model.historian_df['system_external.tm0.test_nm.T_i2'])[1:]) == np.arange(101)[1:] + 1


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_data_with_chunks_no_states(use_llvm, tmpdir):
    external_mappings = []

    data = {'time': np.arange(101),
            'Dew Point Temperature {C}': np.arange(101) + 1,
            'Dry Bulb Temperature {C}': np.arange(101) + 2,
            }

    df = pd.DataFrame(data, columns=['time', 'Dew Point Temperature {C}', 'Dry Bulb Temperature {C}'])
    index_to_timestep_mapping = 'time'
    index_to_timestep_mapping_start = 0
    dataframe_aliases = {
        'system_external.tm0.test_nm.T1': ("Dew Point Temperature {C}", InterpolationType.PIESEWISE),
        'system_external.tm0.test_nm.T2': ('Dry Bulb Temperature {C}', InterpolationType.PIESEWISE)
    }
    path = os.path.join(tmpdir, "x.csv")
    df.to_csv(path)
    external_mappings.append(ExternalMappingElement
                             (path, index_to_timestep_mapping, index_to_timestep_mapping_start, 1,
                              dataframe_aliases))
    data_loader = LocalDataLoader(chunksize=100000)
    s = Simulation(
        Model(StaticDataSystem('system_external', n=1, external_mappings=external_mappings, data_loader=data_loader),
              use_llvm=use_llvm),
        t_start=0, t_stop=100.0, num=100, num_inner=100, max_step=.1)

    s.solve()
    assert approx(np.array(s.model.historian_df['system_external.tm0.test_nm.T_i1'])) == np.arange(101) + 1
    assert approx(np.array(s.model.historian_df['system_external.tm0.test_nm.T_i2'])[1:]) == np.arange(101) + 2


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_data_multiple(use_llvm):
    external_mappings = []
    external_mappings_outer = []

    data = {'time': np.arange(101),
            'Dew Point Temperature {C}': np.arange(101) + 1,
            'Dry Bulb Temperature {C}': np.arange(101) + 2,
            }

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
                               StaticDataSystem('system_external', n=1, external_mappings=external_mappings,
                                                data_loader=data_loader),
                               external_mappings=external_mappings_outer,
                               data_loader=data_loader)
    s = Simulation(
        Model(system_outer, use_llvm=use_llvm),
        t_start=0, t_stop=100.0, num=100, num_inner=100, max_step=.1)
    s.solve()
    assert approx(np.array(s.model.historian_df['system_outer.system_external.tm0.test_nm.T_i1'])) == np.arange(101) + 1
    assert approx(np.array(s.model.historian_df['system_outer.system_external.tm0.test_nm.T_i2'])) == np.arange(101) + 2
    assert approx(np.array(s.model.historian_df['system_outer.tm0.test_nm.T_i1'])) == np.arange(101) + 1
    assert approx(np.array(s.model.historian_df['system_outer.tm0.test_nm.T_i2'])) == np.arange(101) + 2


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


@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_data_chunks_with_states(use_llvm, tmpdir):
    tmax = 5
    external_mappings = []

    data = {'time': np.arange(6),
            'Dew Point Temperature {C}': np.arange(6) + 1,
            }

    df = pd.DataFrame(data, columns=['time', 'Dew Point Temperature {C}'])
    index_to_timestep_mapping = 'time'
    index_to_timestep_mapping_start = 0

    dataframe_aliases = {
        'system_external.tm0.test_nm.T1': ("Dew Point Temperature {C}", InterpolationType.PIESEWISE),
    }
    path = os.path.join(tmpdir, "x.csv")
    df.to_csv(path)
    external_mappings.append(ExternalMappingElement
                             (path, index_to_timestep_mapping, index_to_timestep_mapping_start, 1,
                              dataframe_aliases))
    data_loader = LocalDataLoader(chunksize=3)

    s = Simulation(
        Model(StaticDataSystemWithBall
              ('system_external', n=1, external_mappings=external_mappings, data_loader=data_loader),
              use_llvm=use_llvm),
        t_start=0, t_stop=5, num=100, num_inner=100, max_step=.1)

    s.solve()
    df = s.model.historian_df
    expected_number_of_hits = len(t_hits[t_hits <= tmax])

    df = df[['system_external.ball.t1.t_hit', 'system_external.tm0.test_nm.T_i1']]
    data = df.groupby(df.columns[0]).min().to_records()[1:]
    t_hits_model = [x[0] for x in data]
    interp_model = [x[1] for x in data]
    assert t_hits[:len(t_hits_model)] == approx(t_hits_model, rel=1e-3)  # big accuracy lost may be a problem
    assert [int(t_hit) + 1 for t_hit in t_hits[:len(t_hits_model)]] == approx(interp_model, rel=1e-3)
    assert expected_number_of_hits == len(t_hits_model)
