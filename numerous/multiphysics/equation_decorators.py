# #We can have multiple equation_function in same equation class. Since equation_functions that is decorated as
# differential equations should be treated differently we have two types of decorators.
import uuid
from functools import wraps
import inspect
from textwrap import dedent
from numba import njit


class NumerousFunction(object):
    def __init__(self, signature=None):
        self.id = str(uuid.uuid4())
        self.signature = signature

    def __call__(self, func):
        njited_func = njit(func)
        return njited_func


class Equation(object):

    def __init__(self):
        self.id = str(uuid.uuid4())

    def __call__(self, func):
        @wraps(func)
        def wrapper(self, scope):
            func(self, scope)

        if hasattr(func, '__self__'):
            wrapper.__self__ = func.__self__

        wrapper._equation = True

        wrapper.id = self.id
        try:
            wrapper.lines = inspect.getsource(func)
            wrapper.lineno = inspect.getsourcelines(func)[1]
        except OSError:
            pass
        wrapper.file = inspect.getfile(func)
        wrapper.name = func.__name__
        self.name = func.__name__
        return wrapper


def add_equation(host, func):
    eq = Equation()
    eq_func = eq(func)

    host.equations.append(eq_func)


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

    def __call__(self, func_name, func_source, namespace={}):
        self.name = func_name
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

        func = namespace[func_name]

        @wraps(func)
        def wrapper(self, scope):
            func(self, scope)

        wrapper.name = func_name
        wrapper._equation = True
        wrapper.lines = func_source
        wrapper.id = self.id
        # a = inspect.getsourcelines(func)
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
