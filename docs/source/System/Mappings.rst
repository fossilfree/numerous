Mappings
=============

Mapping is used to connect variables between different ``Item`` and ``Subsystem`` objects within a system.
Mappings are one directional connection that allows for the variables of one object to be used
as input for the equations of another object.
For example, the temperature of one object can be used as input for the heat transfer equation of another object.
The use of mappings allows for the creation of complex systems with a high degree of modularity,
as different items and subsystems can be connected and reused easily.

Explict Mappings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One way to create a mapping between two variables is to use the ``add_mapping``
method on the variable you want to map from. For example:


.. code::

    item_1.namespace_1.x.add_mapping(item_2.namespace_2.y)


This code creates a mapping from ``item_1.namespace_1.y`` to ``item_2.namespace_2.x``,
so that the value of y will be copied into value of x during the simulation.

For example:

.. code::

    class MyItem1(Item,EquationBase):
        def __init__(self, tag='item1'):
            super().__init__(tag)
            namespace = self.create_namespace('t1')
            namespace.add_state('T', 0)
            namespace.add_equation([self])


    class MyItem2(Item,EquationBase):
        def __init__(self, tag='item2'):
            super().__init__(tag)
            namespace = self.create_namespace('t1')
            namespace.add_parameter('Q', 0)
            namespace.add_equation([self])


    item1 = MyItem1()
    item2 = MyItem2()
    item2.t1.Q.add_mapping(item1.t1.T)


In this example, the ``Q`` parameter of ``MyItem2`` is mapped to the ``T`` state of ``MyItem1``.
This means that the value of ``item1.t1.T`` will be used as input (copied before any computation)
for the parameter ``Q``  in the equations of ``MyItem2``.


Mappings with assign and augmented assign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mappings can although be defined using two different types of assignment operator: assign and augmented assign.
Assign operators can replace basic mapping  defined using the add_mapping() method.
This creates a mapping between the two variables, such that the value of x in item A
is assigned to the value of y in item B.
Another way to define mappings is by using augmented assign it will create a mapping that is a sum of
existing value or any other values that was previously sum map.
For example, if we have an item A with a variable x,
and we want to assign the value of x  that is a sum  to another item B's variable y,
we can use the following code:

.. code::

    A.namespace_name.x += B.namespace_name.y

This creates a mapping between the two variables, such that the value of
x in item A assigned to the value of x in item A before any computation of the equation function.

.. note::

    Mappings are only valid within the same namespace and they are only used during the simulation.
    They do not affect the model's state when it's not being solved.



Mappings with connector and ports:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Connector`` is a special type of item that is used to connect
the output variables to input variables of two or more items. Connectors are used to define the relationships between
items in a subsystem. A connector can be created by instantiating the Connector class
from the ``numerous.engine.system.connector`` module.

ConnectorTwoWay is a class in the numerous.engine module of the Numerous Framework for modeling multi-physics systems.
It is a type of Subsystem that represents a connection between two other Subsystems, typically representing fluid flow,
heat transfer, or other physical interactions between two systems.

A ConnectorTwoWay object is used to model two-way interactions between two connected Subsystems.
For example, it can be used to model a pipe connecting two tanks, where the height of liquid in each tank
affects the flow rate of liquid between them.

ConnectorTwoWay objects have two sides, referred to as side1 and side2, which represent the two connected Subsystems.
The two sides have separate namespaces, where variables and equations related to the side can be defined.

To use a ConnectorTwoWay object, you should:

    Define an equation class that represents the interaction between the two sides. This class should inherit
from EquationBase and define equations using the Equation decorator.

    Create a ConnectorTwoWay object, passing in the tag name, and the side1_name and side2_name, which represent
the names of the two sides.

    Create a namespace within the ConnectorTwoWay object and add the equation class to it.

    Create variables in each of the two sides and map them to the corresponding variables in the namespace.
This is necessary to allow the equations to update the values of the variables in the sides.

    Connect the ConnectorTwoWay object to the two connected Subsystems by setting the appropriate
reference to the connector in the Subsystem objects.

Example:

.. code::

    class Spring_Equation(EquationBase):
        def __init__(self, k=1, dx0=1):
            super().__init__(tag='spring_equation')

            self.add_parameter('k', k)
            self.add_parameter('c', 0)
            self.add_parameter('F1', 0)
            self.add_parameter('F2', 0)
            self.add_parameter('x1', 0)
            self.add_parameter('x2', 0)

        @Equation()
        def eval(self, scope):
            ...

    class SpringCoupling(ConnectorTwoWay):
        def __init__(self, tag="springcoup", k=1, dx0=0):
            super().__init__(tag, side1_name='side1', side2_name='side2')

            # 1 Create a namespace for mass flow rate equation and add the valve equation
            mechanics = self.create_namespace('mechanics')
            mechanics.add_equations([Spring_Equation(k=k, dx0=dx0)])

            # 2 Create variables H and mdot in side 1 adn 2
            self.side1.mechanics.create_variable(name='v_dot')
            self.side1.mechanics
