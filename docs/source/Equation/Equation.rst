
Equations in Numerous Engine
==================
Equations in numerous engine is a mathematical expression that describes how the state variables and parameters of a
system change over time. Equations are written as methods on a class that inherits ``EquationBase`` class  from the
``numerous.multiphysics.equation_base`` module. and are decorated with the Equation decorator. These classes are used in
conjunction with the Item and Subsystem classes to simulate the behavior of a system over time. The values of the state
variables and parameters are updated according to the equations and the chosen integration method.

Equation registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create an equation in the Numerous engine, you first need to add any of the following variables to a namespace:

* state: A state variable represents a quantity that changes over time, such as the position or velocity of an object.
In Numerous, state variables are  defined using the ``add_state()`` method. Adding a state variable will automatically
create two variables in the scope object: one for the state and another for
its time-derivative with name ``<state_name>_dot``.

* parameter: A parameter  quantity that can change over time, but is not a state variable.  In Numerous, parameters
are  defined using the ``add_parameter()`` method.

* constant: A variable is a fixed quantity that does not change over time, such as the mass or length of an object.
In Numerous, variables are  defined using the ``add_constant()`` method.

Once the namespace has been created and the variables have been added, you can add an equation to it by calling
the ``add_equation()``
 method on the namespace, passing in the equation object as an argument. For example:

.. code::

    self.<namespace_name>.add_equation(self)


The decorator in the ``numerous.multiphysics.equation_decorators`` module allows users to annotate functions as equations
for use in the Numerous engine. This decorator takes in a function and modifies it to be used as an equation in a system.
The function must take in a scope argument, which provides access to the state, parameter,constants, and global values
of the system.
The function should use the scope object to calculate the time-derivative of the state and store it in a derivatives
named ``<state_name>_dot``.
Here is an example of how to use the Equation decorator to define an equation for a simple system:


.. code::

    class MyItem(Item,EquationBase):
        def __init__(self, tag='my_item'):
            super().__init__(tag)
            # Create a namespace for our equations
            mechanics = self.create_namespace('mechanics')
            # Add variables to the namespace
            mechanics.add_state('x', 0)
            self.mechanics.add_equation(self)

      @Equation()
        def eval(self, scope):
            # Assign value to a derivative of state x
            scope.x_dot = 1



Limitation of equation functions
^^^^^^^^^^^^^^^^^^

It needs to be able to convert the equations into a form that can be efficiently run by a solver.
It is only allowed to use limited set of statements inside an equation function because
In order to  notify that we need to do this, the engine compiles the code in functions decorated with ``@Equation()``.
Augmented assign only support scalar numeric values.
First limitation is that  assign operator can only be used to assign values to variables in a tuple,lists or
a single scalar variable. This have to be accounted if we are using functions
inside equation that are not returning on of mentioned datatypes. The following example shows use of assign operator:

.. code::

    @Equation()
    def eval(self, scope):

        # Assign values to tuple of variables
        scope.x, scope.y, scope.z = (1, 2, 3)
        # Assign values to list variable
        my_list = [4, 5, 6]
        # Using subscript to access list value
        scope.f = my_list[0]
        # Assign values to set variable
        my_set = {7, 8, 9}
        # Using subscript to access set value
        scope.q = list(my_set)[0]


Another important limitation of equations inside numerous engine is not full support of ``if statements``
and if expressions.
We are not allowed to use nested ``if statements`` and only
scalar variables are allowed to be compared in ``if statement``.

One way to avoid such limitations is to write complex functions outside of the equation body
and compile it using ``njit`` decorator or Numerous function decorator form numerous engine.
There couple of ways how we can add such external functions to the equitation body.

Closure inside the item class
----------------

Closure inside the item class

Imported from external library
----------------

Imported from external library

NumerousFunction decorator
----------------

NumerousFunction decorator

Global variables inside equation method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to use global variables inside the equation decorated method.
There is one pre-defined global variable ``t``  in equation that is time variable that allow as to accesses
current time that is used by the solver.
To add another global variable to be used inside equation we have to import them separately
on a model level.
To use global variables inside the equation we can access them using ``global_vars`` key inside
the scope that is passed to the equation annotated method.

.. note::

    It is not possible to assign to ``global_vars`` variables.

.. code::

    @Equation()
    def eval(self, scope):
        scope.T = scope.global_vars.constant_value
