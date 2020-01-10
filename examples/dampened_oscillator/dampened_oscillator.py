from numerous.multiphysics.equation_decorators import equation
from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.engine.system import Subsystem

class DampenedOscillator(EquationBase, Item):
    """
        Equation and item modelling a spring and dampener
    """
    def __init__(self, tag="tm", x0=1, k=1, c=1):
        super(DampenedOscillator, self).__init__(tag)

        #define variables
        self.add_constant('k', k)
        self.add_constant('c', c)
        self.add_state('x', x0)
        self.add_state('v', 0)

        #define namespace and add equation
        mechanics = self.create_namespace('mechanics')
        mechanics.add_equations([self])

    @Equation()
    def eval(self, scope):
        #Implement equations for the dampened oscillation
       scope.v_dot = -scope.k * scope.x - scope.c * scope.v
       scope.x_dot = scope.v

class OscillatorSystem(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)

        #Create oscillator
        oscillator = DampenedOscillator('oscillator', k=1, c=1, x0=10)

        #Register the items to the subsystem to make it recognize them.
        self.register_items([oscillator])




