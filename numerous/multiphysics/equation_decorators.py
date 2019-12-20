# #We can have multiple equation_function in same equation class. Since equation_functions that is decorated as
# differential equations should be treated differently we have two types of decorators.


def differential_equation(func):
    def wrapper_(equation_function, scope):
        scope.apply_differential_equation_rules(True)
        func(equation_function, scope)
        scope.apply_differential_equation_rules(False)

    wrapper_._equation = True
    return wrapper_

def equation(func):
    def wrapper_(equation_function, scope):
        func(equation_function, scope)

    wrapper_._equation = True
    return wrapper_
