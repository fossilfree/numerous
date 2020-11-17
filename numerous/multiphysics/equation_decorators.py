# #We can have multiple equation_function in same equation class. Since equation_functions that is decorated as
# differential equations should be treated differently we have two types of decorators.
import uuid
from functools import wraps
import inspect


class Equation(object):

    def __init__(self):
        self.id =str(uuid.uuid1())
    # self.func = func

    def __call__(self, func):
        @wraps(func)
        def wrapper(self,scope):
            func(self,scope)
        wrapper._equation = True
        wrapper.lines = inspect.getsource(func)
        wrapper.id = self.id
        #a = inspect.getsourcelines(func)
        wrapper.lineno = inspect.getsourcelines(func)[1]
        wrapper.file = inspect.getfile(func)
        # wrapper.i = self.i
        return wrapper


class DifferentialEquation(Equation):

    def __init__(self, func):
        super().__init__(func)

    def __call__(self, scope):
        scope.apply_differential_equation_rules(True)
        self.func(scope)
        scope.apply_differential_equation_rules(False)
