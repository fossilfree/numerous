from numerous.engine.model import Model
from numerous.engine.simulation import Simulation

from numerous.engine.system import Subsystem, ConnectorItem, Item, ConnectorTwoWay
from numerous import EquationBase, Equation
from numerous.engine.simulation.solvers.base_solver import  SolverType
from tests.test_equations import TestEq_ground, Test_Eq, TestEq_input




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
import os.path
model_filename = "./export_model/S3.numerous"
if not os.path.isfile(model_filename):
    Ms_3 = S3('S3')
    m1 = Model(Ms_3, use_llvm=True,export_model=True)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100, solver_type=SolverType.NUMEROUS)
    s1.solve()
    print(list(m1.states_as_vector[::-1]))

else:
    m1 = Model.from_file(model_filename)
    s1 = Simulation(m1, t_start=0, t_stop=1000, num=100, solver_type=SolverType.NUMEROUS)
    s1.solve()
    print(list(m1.states_as_vector[::-1]))
# [2010, 1010, 510, 210]