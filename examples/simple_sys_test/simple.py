from numerous.engine.system import Subsystem, ItemsStructure
from numerous.engine.system.item import Item
from numerous.multiphysics.equation_base import EquationBase
from numerous.multiphysics.equation_decorators import Equation

if __name__ == "__main__":
    from numerous.engine import model, simulation


class Simple(EquationBase, Item):
    """
        Equation and item modelling a spring and dampener
    """

    def __init__(self, tag="simple", x0=1, k=1):
        super(Simple, self).__init__(tag)

        # define variables
        self.add_constant('k', k)

        self.add_state('x', x0)


        # define namespace and add equation
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):

        scope.x_dot = scope.k + 0*scope.x


class SimpleSystem(Subsystem):
    def __init__(self, tag, k=.1, n=1, x0=[0]):
        super().__init__(tag)
        simples = []
        for i in range(n):
            # Create oscillator
            simple = Simple('simple' + str(i), k=k*(i+1), x0=x0[i])
            simples.append(simple)

        self.register_items(simples, tag="simples", structure=ItemsStructure.LIST)



if __name__ == "__main__":
    from numerous.engine import model, simulation

    subsystem = SimpleSystem('system', k=.1, n=2, x0=[0, 0])
    # Define simulation
    s = simulation.Simulation(model.Model(subsystem), t_start=0, t_stop=1, num=100, num_inner=100, max_step=1,
                              use_llvm=True, save_to_file=True)
