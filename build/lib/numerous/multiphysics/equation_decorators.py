# #We can have multiple equation_function in same equation class. Since equation_functions that is decorated as
# differential equations should be treated differently we have two types of decorators.
from functools import wraps


class Equation(object):

    # def __init__(self):
    # self._equation = True
    # self.func = func

    def __call__(self, func):
        @wraps(func)
        def wrapper(self,scope):
            func(self,scope)

        wrapper._equation = True
        return wrapper


class DifferentialEquation(Equation):

    def __init__(self, func):
        super().__init__(func)

    def __call__(self, scope):
        scope.apply_differential_equation_rules(True)
        self.func(scope)
        scope.apply_differential_equation_rules(False)
