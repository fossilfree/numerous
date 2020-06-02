from numerous import EquationBase, Equation


class TestEq_input(EquationBase):
    def __init__(self, P=10, T=0, R=1):
        super().__init__(tag='input_eq')
        self.add_parameter('P', P)
        self.add_parameter('T_o', 0)
        self.add_state('T', T)
        self.add_constant('R', R)

    @Equation()
    def eval(self,scope):
        scope.T_dot = scope.P - (scope.T - scope.T_o) / scope.R


class Test_Eq(EquationBase):
    def __init__(self, T=0, R=1):
        super().__init__(tag='T_eq')
        self.add_state('T', T)
        self.add_parameter('R_i', 0)
        self.add_parameter('T_i', 0)
        self.add_parameter('T_o', 0)
        self.add_constant('R', R)

    @Equation()
    def eval(self, scope):
        scope.T_dot = (scope.T_i - scope.T) / scope.R_i - (scope.T - scope.T_o) / scope.R



class TestEq_ground(EquationBase):
    def __init__(self, TG=10, RG=2):
        super().__init__(tag='ground_eq')
        self.add_constant('T', TG)
        self.add_constant('R', RG)


class TestEq_dictState(EquationBase):
    def __init__(self):
        super().__init__(tag='ground_eq')
        self.add_state('T', {})



