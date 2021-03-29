# #We can have multiple equation_function in same equation class. Since equation_functions that is decorated as
# differential equations should be treated differently we have two types of decorators.
import uuid
from functools import wraps
import inspect
from textwrap import dedent


class Equation(object):

    def __init__(self):
        self.id =str(uuid.uuid4())
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

def dedent_code(code):
    tries = 0
    while tries < 10:
        try:
            dsource = dedent(dsource)

            break
        except IndentationError:

            tries += 1
            if tries > 10 - 1:
                print(dsource)
                raise

class InlineEquation(Equation):

    def __call__(self, func_name, func_source, namespace = {}):

        tries = 0
        while tries < 10:
            try:
                func_source = dedent(func_source)
                exec(func_source, namespace)
                break
            except IndentationError:

                tries += 1
                if tries > 10 - 1:
                    print(func_source)
                    raise

        print(func_source)
        func = namespace[func_name]
        @wraps(func)
        def wrapper(self,scope):
            func(self, scope)

        wrapper._equation = True
        wrapper.lines = func_source
        wrapper.id = self.id
        #a = inspect.getsourcelines(func)
        wrapper.lineno = 0
        wrapper.file = 'dynamic.py'
        # wrapper.i = self.i
        return wrapper


class DifferentialEquation(Equation):

    def __init__(self, func):
        super().__init__(func)

    def __call__(self, scope):
        scope.apply_differential_equation_rules(True)
        self.func(scope)
        scope.apply_differential_equation_rules(False)
