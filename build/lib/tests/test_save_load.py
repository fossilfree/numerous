import os

from pytest import approx
import pytest

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.simulation.simulation_callbacks import _SimulationCallback
from numerous.engine.system import ConnectorItem, ConnectorTwoWay, Item, Subsystem
from numerous import HistoryDataFrame
from .test_equations import TestEq_input, Test_Eq, TestEq_ground


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


@pytest.fixture()
def filename():
    filename  = 'test_stop_start'
    yield filename
    os.remove(filename)



def test_start_stop_model(filename):
    def stop_callback(t, _):
        if t > 1:
            raise ValueError("Overflow of state2")

    hdf = HistoryDataFrame()
    m1 = Model(S3('S3'), historian=hdf)

    c1 = _SimulationCallback("test")
    c1.add_callback_function(stop_callback)
    m1.callbacks.append(c1)
    m1.save_variables_schedule(0.1, filename)

    s1 = Simulation(m1, t_start=0, t_stop=2, num=100)
    with pytest.raises(ValueError, match=r".*Overflow of state2.*"):
        s1.solve()
    hdf2 = HistoryDataFrame.load(filename)
    m2 = Model(S3('S3'), historian=hdf2)
    m2.restore_state()

    assert approx(m2.states_as_vector, rel=0.1) == [89.8, 3.9, 0.7, 3.3]

    s2 = Simulation(m2, t_start=0, t_stop=1, num=50)
    s2.solve()
    m3 = Model(S3('S3'), historian=hdf)

    s3 = Simulation(m3, t_start=0, t_stop=2, num=100)
    s3.solve()
    print(m3.info)
    assert approx(m3.states_as_vector, rel=0.1) == m2.states_as_vector
