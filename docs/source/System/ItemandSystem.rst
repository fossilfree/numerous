
Item
==================

Main modeling element of the  numerous engine is the ``Item`` class, which is used to define
a single simulation element.

Namespaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``Item`` object can contain one or more ``namespaces``, which are used to organize the equations and variables for that component.
Namespace represents a specific physical domain or area of the system, and can contain its own set of equations and variables.
Each namespace can contain one or more ``Equation`` objects, which are used to define the mathematical
relationships between the variables in the namespace.
Namespaces are created by calling the ``create_namespace()`` method on an Item or Subsystem object. For example:

.. code::

    class MyItem(Item):
        def __init__(self, tag='my_item'):
            super().__init__(tag)
            # Create a namespace for our equations
            mechanics = self.create_namespace('mechanics')
            # Add variables to the namespace
            mechanics.add_state('x', 0)

In the above example, a namespace called ``mechanics`` is created on the MyItem object, and a state variable x is added to it.
Once a namespace is created, equations can be added to it by calling the ``add_equations()``
method on the namespace and passing in a list of equation objects as an argument.

.. code::

    class MyItem(Item,EquationBase):
        def __init__(self, tag='my_item'):
            super().__init__(tag)
            mechanics = self.create_namespace('mechanics')
            mechanics.add_state('x', 0)
            mechanics.add_equations([self])
       @Equation()
        def eval_(self, scope):
            scope.x_dot = 1
