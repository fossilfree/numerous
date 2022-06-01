import os
import time
from enum import Enum

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation

from numerous.engine.system import Subsystem, Item
from numerous.engine.simulation.solvers.base_solver import SolverType
from tests.test_equations import TestEq_ground, Test_Eq, TestEq_input


class SolverType(Enum):
    SOLVER_IVP = 0
    NUMEROUS = 1


solver_types = [SolverType.NUMEROUS, SolverType.SOLVER_IVP]


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


model_filename = "./export_model/S2.numerous"
import sys

sys.setrecursionlimit(100000)
if not os.path.isfile(model_filename):
    Ms_3 = S2N("S2", 1000)
    m1 = Model(Ms_3, use_llvm=True, export_model=True)

    start = time.time()
    Ms_4 = S2N("S2", 1000)
    m2 = Model(Ms_4, use_llvm=True, export_model=False)
    end = time.time()
    # 4.
    print("model compilation time ", end - start)

    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100)
    s1.solve()
    print(list(m1.states_as_vector[::-1]))
else:
    start = time.time()
    m1 = Model.from_file(model_filename)
    end = time.time()
    print("model from file  ", end - start)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100)
    s1.solve()
    print(list(m1.states_as_vector[::-1]))
