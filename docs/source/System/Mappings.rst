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
Assign operators can replace basic mapping  defined using the ``add_mapping()`` method.
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



Mappings with connector and ports
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Connector`` is a special type of item that is used to connect
the output variables to input variables of two or more items. Connectors are used to define the relationships between
items in a subsystem. A connector can be created by instantiating the Connector class
from the ``numerous.engine.system.connector`` module. The `` Connector``  allows for the creation of a typical connection before defining the subsystems that will be connected.
A ``ConnectorTwoWay`` object is used to model two-way interactions between two connected Subsystems.

``ConnectorTwoWay`` objects have two predefined sides, referred to as ``side1`` and ``side2``,
which represent the two  Subsystems or Items.
The two sides have separate namespaces, where variables related to the side can be defined.

To use a ``ConnectorTwoWay`` object, you should:

#. Create a namespace within the ConnectorTwoWay object and add the equation class to it or create each individual variable expected in this namespace.
#. Use assign operator mappings to map variables between two sides.
#. Use bind(side1,side2) method of the ``ConnectorTwoWay`` after it is instantiated to specify the systems we are connecting.


Example:

.. code::

        class T(ConnectorTwoWay):
        def __init__(self, tag, T, R):
            super().__init__(tag, side1_name='input', side2_name='output')

            t1 = self.create_namespace('t1')
            t1.add_equations([Test_Eq(T=T, R=R)])

            t1.R_i = self.input.t1.R
            t1.T_i = self.input.t1.T

            ##we ask for variable T
            t1.T_o = self.output.t1.T

    class S3(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)

            input = I('1', P=100, T=0, R=10)
            item1 = T('2', T=0, R=5)
            item2 = T('3', T=0, R=3)
            item3 = T('4', T=0, R=2)
            ## RG is redundant we use item3.R as a last value of R in a chain
            ground = G('5', TG=10, RG=2)

            input.bind(output=item1)

            item1.bind(input=input, output=item2)