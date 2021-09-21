from abc import ABC

import pytest
from numerous.engine.model.external_mappings import ExternalMappingElement

from numerous.utils.data_loader import InMemoryDataLoader
from pytest import approx

from numerous.engine.model.external_mappings.interpolation_type import InterpolationType
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation

from numerous.engine.system import Subsystem, ConnectorItem, Item, ConnectorTwoWay
from numerous import EquationBase, Equation
from numerous.engine.simulation.solvers.base_solver import solver_types, SolverType
from tests.test_equations import TestEq_ground, Test_Eq, TestEq_input


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    import shutil
    shutil.rmtree('./tmp', ignore_errors=True)
    yield


@pytest.fixture
def test_eq1():
    class TestEq1(EquationBase):
        def __init__(self, P=10):
            super().__init__(tag='example_1')
            self.add_parameter('P', P)
            self.add_state('T1', 0)
            self.add_state('T2', 0)
            self.add_state('T3', 0)
            self.add_state('T4', 0)
            # self.add_parameter('T_4', 0)
            self.add_constant('TG', 10)
            self.add_constant('R1', 10)
            self.add_constant('R2', 5)
            self.add_constant('R3', 3)
            self.add_constant('RG', 2)

        @Equation()
        def eval(self, scope):
            scope.T1_dot = scope.P - (scope.T1 - scope.T2) / scope.R1
            scope.T2_dot = (scope.T1 - scope.T2) / scope.R1 - (scope.T2 - scope.T3) / scope.R2
            scope.T3_dot = (scope.T2 - scope.T3) / scope.R2 - (scope.T3 - scope.T4) / scope.R3
            scope.T4_dot = (scope.T3 - scope.T4) / scope.R3 - (scope.T4 - scope.TG) / scope.RG

    return TestEq1(P=100)


@pytest.fixture
def simple_item(test_eq1):
    class T1(Item):
        def __init__(self, tag):
            super().__init__(tag)

            t1 = self.create_namespace('t1')

            t1.add_equations([test_eq1])

    return T1('test_item')


@pytest.fixture
def ms1(simple_item):
    class S1(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            self.register_items([simple_item])

    return S1('S1')


@pytest.fixture
def ms2():
    class I(Item):
        def __init__(self, tag, P, T, R):
            super().__init__(tag)

            t1 = self.create_namespace('t1')
            t1.add_equations([TestEq_input(P=P, T=T, R=R)])

    class T(Item):
        def __init__(self, tag, T, R):
            super().__init__(tag)

            t1 = self.create_namespace('t1')
            t1.add_equations([Test_Eq(T=T, R=R)])

    class G(Item):
        def __init__(self, tag, TG, RG):
            super().__init__(tag)

            t1 = self.create_namespace('t1')
            t1.add_equations([TestEq_ground(TG=TG, RG=RG)])

    class S2(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)

            input = I('1', P=100, T=0, R=10)
            item1 = T('2', T=0, R=5)
            item2 = T('3', T=0, R=3)
            item3 = T('4', T=0, R=2)
            ## RG is redundant we use item3.R as a last value of R in a chain
            ground = G('5', TG=10, RG=2)

            input.t1.T_o.add_mapping(item1.t1.T)

            # item1.bind(input=input, output=item2)

            item1.t1.R_i.add_mapping(input.t1.R)
            item1.t1.T_i.add_mapping(input.t1.T)
            item1.t1.T_o.add_mapping(item2.t1.T)
            #  t_0 = item1.t1.T_o
            # item1.t1.T_o = item2.t1.T

            item2.t1.R_i.add_mapping(item1.t1.R)
            item2.t1.T_i.add_mapping(item1.t1.T)
            item2.t1.T_o.add_mapping(item3.t1.T)

            item3.t1.R_i.add_mapping(item2.t1.R)
            item3.t1.T_i.add_mapping(item2.t1.T)
            item3.t1.T_o.add_mapping(ground.t1.T)

            self.register_items([input, item1, item2, item3, ground])

    return S2('S2')


@pytest.fixture
def ms3():
    class I(ConnectorItem):
        def __init__(self, tag, P, T, R):
            super(I, self).__init__(tag)

            self.create_binding('output')

            t1 = self.create_namespace('t1')

            t1.add_equations([TestEq_input(P=P, T=T, R=R)])
            ##this line has to be after t1.add_equations since t1 inside output is created there
            self.output.t1.create_variable(name='T')
            t1.T_o = self.output.t1.T

    class T(ConnectorTwoWay):
        def __init__(self, tag, T, R):
            super().__init__(tag, side1_name='input', side2_name='output')

            t1 = self.create_namespace('t1')
            t1.add_equations([Test_Eq(T=T, R=R)])

            t1.R_i = self.input.t1.R
            t1.T_i = self.input.t1.T

            ##we ask for variable T
            t1.T_o = self.output.t1.T

    class G(Item):
        def __init__(self, tag, TG, RG):
            super().__init__(tag)

            t1 = self.create_namespace('t1')
            t1.add_equations([TestEq_ground(TG=TG, RG=RG)])

            # ##since we asked for variable T in binding we have to create variable T and map it to TG
            # t1.create_variable('T')
            # t1.T = t1.TG

    class S3(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)

            input = I('1', P=100, T=0, R=10)
            item1 = T('2', T=0, R=5)
            item2 = T('3', T=0, R=3)
            item3 = T('4', T=0, R=2)
            ## RG is redundant we use item3.R as a last value of R in a chain
            ground = G('5', TG=10, RG=2)

            input.bind(output=item1)

            item1.bind(input=input, output=item2)

            item2.bind(input=item1, output=item3)
            item3.bind(input=item2, output=ground)

            self.register_items([input, item1, item2, item3, ground])

    return S3('S3')


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_model_var_referencing(ms1, solver, use_llvm):
    m1 = Model(ms1, use_llvm=use_llvm)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10, solver_type=solver)
    s1.solve()
    assert approx(list(m1.states_as_vector[::-1]), rel=0.01) == [2010, 1010, 510, 210]


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.skip(reason="Functionality not implemented in current version")
def test_model_save_only_aliases(ms3, solver):
    of = OutputFilter(only_aliases=True)
    m1 = Model(ms3, historian_filter=of)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10, solver_type=solver)
    s1.solve()
    assert m1.historian_df.empty


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.skip(reason="Functionality not implemented in current version")
def test_model_save_only_aliases2(ms3, solver):
    of = OutputFilter(only_aliases=True)
    m1 = Model(ms3, historian_filter=of)
    item = m1.search_items('2')[0]
    columns_number = 0
    for i, var in enumerate(item.get_variables()):
        var[0].alias = str(i)
        columns_number += 1

    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10, solver_type=solver)
    s1.solve()
    assert m1.historian_df.columns.size == columns_number


def test_1_item_model(ms1):
    m1 = Model(ms1)
    item = m1.search_items('test_item')[0]
    assert item.t1.P.value == 100


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_callback_step_item_model(ms3, solver, use_llvm):
    def action(time, variables):
        if int(time) == 119:
            raise ValueError("Overflow of state. time:119")

    def condition(time, states):
        return 500 - states['S3.3.t1.T']

    def action2(time, variables):
        if int(time) == 118:
            raise ValueError("Overflow of state. time:119")

    def condition2(time, states):
        return 500 - states['S3.3.t1.T']

    m1 = Model(ms3, use_llvm=use_llvm)
    m1.add_event("simple", condition, action)
    m1.add_event("simple2", condition2, action2)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100, solver_type=solver)
    with pytest.raises(ValueError, match=r".*time:119.*"):
        s1.solve()


def test_add_item_twice_with_same_tag(ms2):
    class Item_(Item):
        def __init__(self, tag):
            super().__init__(tag)

    with pytest.raises(ValueError, match=r".*already registered in system.*"):
        ms2.register_items([Item_('1')])


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_chain_item_model(ms2, solver, use_llvm):
    m1 = Model(ms2, use_llvm=use_llvm)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10, solver_type=solver)
    s1.solve()
    assert approx(m1.states_as_vector, rel=0.01) == [2010, 1010, 510, 210]


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_chain_item_binding_model_nested(ms3, solver, use_llvm):
    ms4 = Subsystem('new_s')
    ms4.register_item(ms3)
    m1 = Model(ms4, use_llvm=use_llvm)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10, solver_type=solver)
    s1.solve()
    assert approx(m1.states_as_vector, rel=0.01) == [2010, 1010, 510, 210]


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_chain_item_binding_model_nested2(ms3, solver, use_llvm):
    ms4 = Subsystem('new_s4')
    ms4.register_item(ms3)
    ms5 = Subsystem('new_s5')
    ms5.register_item(ms3)
    ms6 = Subsystem('new_s6')
    ms6.register_item(ms4)
    ms6.register_item(ms5)
    ms7 = Subsystem('new_s7')
    ms7.register_item(ms6)
    m1 = Model(ms7, use_llvm=use_llvm)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100, solver_type=solver)
    s1.solve()
    assert len(m1.path_variables) == 50
    assert len(m1.variables) == 25
    assert approx(m1.states_as_vector, rel=0.01) == [2010, 1010, 510, 210]


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_chain_item_binding_model(ms3, solver, use_llvm):
    m1 = Model(ms3, use_llvm=use_llvm)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100, solver_type=solver)
    s1.solve()
    assert approx(m1.states_as_vector, rel=0.01) == [2010, 1010, 510, 210]


class StaticDataTest(EquationBase, Item):
    def __init__(self, tag="tm"):
        super(StaticDataTest, self).__init__(tag)

        ##will map to variable with the same path in external dataframe/datasource
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
    def __init__(self, tag, n=1):
        super().__init__(tag)
        o_s = []
        for i in range(n):
            o = StaticDataTest('tm' + str(i))
            o_s.append(o)
        # Register the items to the subsystem to make it recognize them.
        self.register_items(o_s)


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_external_data(solver, use_llvm):
    external_mappings = []

    import pandas as pd
    import numpy as np

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
    s = Simulation(
        Model(StaticDataSystem('system_external', n=1), use_llvm=use_llvm, external_mappings=external_mappings,
              data_loader=data_loader),
        t_start=0, t_stop=100.0, num=100, num_inner=100, max_step=.1, solver_type=solver
    )
    s.solve()
    assert approx(np.array(s.model.historian_df['system_external.tm0.test_nm.T_i1'])[1:]) == np.arange(101)[1:]
    assert approx(np.array(s.model.historian_df['system_external.tm0.test_nm.T_i2'])[1:]) == np.arange(101)[1:] + 1


@pytest.mark.parametrize("solver", solver_types)
@pytest.mark.parametrize("use_llvm", [True, False])
def test_static_system(solver, use_llvm):
    import numpy as np
    s = Simulation(
        Model(StaticDataSystem('system_static', n=1), use_llvm=use_llvm),
        t_start=0, t_stop=100.0, num=100, num_inner=100, max_step=.1, solver_type=solver
    )
    s.solve()
    assert approx(np.array(s.model.historian_df['system_static.tm0.test_nm.T_i1'])[1:]) == np.repeat(0, (100))
    assert approx(np.array(s.model.historian_df['system_static.tm0.test_nm.T_i2'])[1:]) == np.repeat(0, (100))
