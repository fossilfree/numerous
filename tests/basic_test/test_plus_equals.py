from numerous.multiphysics.equation_decorators import Equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation
import pytest


class TestEQ(EquationBase, Item):
    """
        Equation and item modelling a spring and dampener
    """

    def __init__(self, tag="test_eq", x0=1, k=1):
        super(TestEQ, self).__init__(tag)
        # define variables
        self.add_constant('k', k)
        self.add_parameter('p', k)
        self.add_state('x', x0)
        # define namespace and add equation
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        p = 1
        p += scope.k
        scope.p = p


class TestSystem(Subsystem):
    def __init__(self, k, x, tag="test_sys"):
        super().__init__(tag)
        self.register_item(TestEQ(k=k, x0=x))


@pytest.mark.parametrize("use_llvm", [True, False])
def test_augmented_assign(use_llvm):
    k = 1
    x = 1
    test_system = TestSystem(k=k, x=x)
    test_model = Model(test_system, use_llvm=use_llvm, save_to_file=True)

    test_simulation = Simulation(test_model, solver_type=SolverType.NUMEROUS, t_start=0, t_stop=100, num=10,
                                 num_inner=1, max_step=1)
    test_simulation.solve()

    result = test_simulation.model.historian_df["test_sys.test_eq.mechanics.p"]
    print(result)
    assert list(result)[0] == 2.0
