import pytest

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Item, Subsystem
from tests.test_equations import TestEq_input, Test_Eq, TestEq_ground


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


class S2N(Subsystem):
    def __init__(self, tag, n):
        super().__init__(tag)
        items = []
        input = I('1', P=100, T=0, R=10)
        for i in range(n):
            items.append(T(str(i + 2), T=1, R=5))
        ground = G(str(n + 2), TG=10, RG=2)

        input.t1.T_o.add_mapping(items[0].t1.T)

        for item in range(n):
            if item == 0:
                items[item].t1.R_i.add_mapping(input.t1.R)
                items[item].t1.T_o.add_mapping(items[item + 1].t1.T)
            elif item == n - 1:
                items[item].t1.R_i.add_mapping(items[item - 1].t1.R)
                items[item].t1.T_i.add_mapping(items[item - 1].t1.T)
                items[item].t1.T_o.add_mapping(ground.t1.T)
            else:
                items[item].t1.R_i.add_mapping(items[item - 1].t1.R)
                items[item].t1.T_i.add_mapping(items[item - 1].t1.T)
                items[item].t1.T_o.add_mapping(items[item + 1].t1.T)

        r_items = [input]
        for i in items:
            r_items.append(i)
        r_items.append(ground)
        self.register_items(r_items)


def test_clone_model():
    system = S2N("S2", 2)
    model = Model(system, clonable=True)
    s1 = Simulation(model, t_start=0, t_stop=2, num=2)
    s1.solve()
    s2 = Simulation(model.clone(), t_start=0, t_stop=2, num=2)
    s2.solve()
    assert all(s1.model.states_as_vector == s2.model.states_as_vector)


def test_clone_model_multiple():
    system = S2N("S2", 2)
    model = Model(system, clonable=True)
    s1 = Simulation(model, t_start=0, t_stop=2, num=2)
    s1.solve()
    s2 = Simulation(model.clone(clonable=True).clone(), t_start=0, t_stop=2, num=2)
    s2.solve()
    assert all(s1.model.states_as_vector == s2.model.states_as_vector)


def test_clone_exception():
    with pytest.raises(Exception, match="Model isn't clonable"):
        system = S2N("S2", 2)
        model = Model(system)
        model.clone()


