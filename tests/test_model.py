import pytest
from pytest import approx

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.simulation.simulation_callbacks import _SimulationCallback
from numerous.engine.system import Subsystem, ConnectorItem, Item, ConnectorTwoWay
from numerous import EquationBase, HistoryDataFrame, OutputFilter, Equation
from .test_equations import TestEq_ground, Test_Eq, TestEq_input


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
            self.add_parameter('T_4', {})
            self.add_constant('TG', 10)
            self.add_constant('R1', 10)
            self.add_constant('R2', 5)
            self.add_constant('R3', 3)
            self.add_constant('RG', 2)

        @Equation()
        def eval(self,scope):
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


def test_model_var_referencing(ms1):
    m1 = Model(ms1)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10)
    s1.solve()
    assert approx(m1.states_as_vector[::-1], rel=0.01) == [2010, 1010, 510, 210]


def test_model_save_only_aliases(ms3):
    hdf = HistoryDataFrame(filter=OutputFilter(only_aliases=True))
    m1 = Model(ms3, historian=hdf)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10)
    s1.solve()
    assert hdf.df.empty


def test_model_save_only_aliases2(ms3):
    hdf = HistoryDataFrame(filter=OutputFilter(only_aliases=True))
    m1 = Model(ms3, historian=hdf)
    item = m1.search_items('2')[0]
    columns_number = 0
    for i, var in enumerate(item.get_variables()):
        var[0].alias = str(i)
        columns_number += 1

    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10)
    s1.solve()
    assert hdf.df.columns.size == columns_number


def test_1_item_model(ms1):
    m1 = Model(ms1)
    item = m1.search_items('test_item')[0]
    assert item.t1.P.value == 100


def test_callback_step_item_model(ms1):
    def simple_callback(_, variables):
        if variables['S1.test_item.t1.T2'].value > 1000:
            raise ValueError("Overflow of state2")

    m1 = Model(ms1)
    c1 = _SimulationCallback("test")
    c1.add_callback_function(simple_callback)
    m1.callbacks.append(c1)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10)
    with pytest.raises(ValueError, match=r".*Overflow of state2.*"):
        s1.solve()


def test_add_item_twice_with_same_tag(ms2):
    class Item_(Item):
        def __init__(self, tag):
            super().__init__(tag)

    with pytest.raises(ValueError, match=r".*already registered in system.*"):
        ms2.register_items([Item_('1')])


def test_chain_item_model(ms2):
    m1 = Model(ms2)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100)
    s1.solve()
    assert approx(m1.states_as_vector, rel=0.01) == [2010, 1010, 510, 210]


def test_chain_item_binding_model(ms3):
    m1 = Model(ms3)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100)
    s1.solve()
    assert approx(m1.states_as_vector, rel=0.01) == [2010, 1010, 510, 210]
