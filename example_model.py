
from enum import Enum

from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
from numerous.engine.system import Subsystem, Item
from tests.test_equations import TestEq_ground, Test_Eq, TestEq_input


class SolverType(Enum):
    SOLVER_IVP = 0
    NUMEROUS = 1


solver_types = [SolverType.NUMEROUS, SolverType.SOLVER_IVP]

#Input item supplies the system with heat
class I(Item):
    def __init__(self, tag, P, T, R):
        super().__init__(tag)

        t1 = self.create_namespace('t1')
        t1.add_equations([TestEq_input(P=P, T=T, R=R)])

#T item is the middle of the system and transfers heat to its neighbors
class T(Item):
    def __init__(self, tag, T, R):
        super().__init__(tag)

        t1 = self.create_namespace('t1')
        t1.add_equations([Test_Eq(T=T, R=R)])

#G item is the exit of the system
class G(Item):
    def __init__(self, tag, TG, RG):
        super().__init__(tag)

        t1 = self.create_namespace('t1')
        t1.add_equations([TestEq_ground(TG=TG, RG=RG)])

#Non-scaling example
def ms2():
    class S2(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)

            #Register all of the items
            input = I('1', P=100, T=0, R=10)
            item1 = T('2', T=0, R=5)
            item2 = T('3', T=0, R=3)
            item3 = T('4', T=0, R=2)
            ## RG is redundant we use item3.R as a last value of R in a chain
            ground = G('5', TG=10, RG=2)

            #add item mappings
            input.t1.T_o.add_mapping(item1.t1.T)

            item1.t1.R_i.add_mapping(input.t1.R)
            item1.t1.T_i.add_mapping(input.t1.T)
            item1.t1.T_o.add_mapping(item2.t1.T)

            item2.t1.R_i.add_mapping(item1.t1.R)
            item2.t1.T_i.add_mapping(item1.t1.T)
            item2.t1.T_o.add_mapping(item3.t1.T)

            item3.t1.R_i.add_mapping(item2.t1.R)
            item3.t1.T_i.add_mapping(item2.t1.T)
            item3.t1.T_o.add_mapping(ground.t1.T)

            #Add items to system
            self.register_items([input, item1, item2, item3, ground])

    return S2('S2')

#general function to run a model simulation
def run_model(ms, solver, use_llvm):
    # print(type(ms2))
    m1 = Model(system=ms, use_llvm=use_llvm)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=10, solver_type=solver)
    s1.solve()
    return s1

#Scaling example
def ms2N(n):
    class S2N(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            items = []

            #Register input and ground as usual, but add T items into a list
            input = I('1', P=100, T=0, R=10)
            for i in range(n):
                items.append(T(str(i+2), T=1, R=5))
            ground = G(str(n + 2), TG=10, RG=2)

            input.t1.T_o.add_mapping(items[0].t1.T)

            #Add mappings
            for item in range(n):
                #account for special case when item is the first
                if item == 0:
                    items[item].t1.R_i.add_mapping(input.t1.R)
                    items[item].t1.T_o.add_mapping(items[item + 1].t1.T)
                #account for special case when item is the last
                elif item == n-1:
                    items[item].t1.R_i.add_mapping(items[item - 1].t1.R)
                    items[item].t1.T_i.add_mapping(items[item - 1].t1.T)
                    items[item].t1.T_o.add_mapping(ground.t1.T)
                else:
                    items[item].t1.R_i.add_mapping(items[item - 1].t1.R)
                    items[item].t1.T_i.add_mapping(items[item - 1].t1.T)
                    items[item].t1.T_o.add_mapping(items[item + 1].t1.T)

            #place all items in a single list and register them into the system
            r_items = [input]
            for i in items:
                r_items.append(i)
            r_items.append(ground)
            self.register_items(r_items)

    return S2N('S2')


s1n=run_model(ms2N(5), solver=SolverType.NUMEROUS, use_llvm=[True, False])
s1n_result=s1n.model.historian_df
