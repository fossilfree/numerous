Mappings
=============

Mapping is used to connect variables between different ``Item`` and ``Subsystem`` objects within a system.
This allows for the variables of one object to be used as input for the equations of another object.
For example, the temperature of one object can be used as input for the heat transfer equation of another object.

Explict Mappings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One way to create a mapping between two variables, you can use the ``add_mapping``
method on the variable you want to map from. For example:


.. code::

    item_1.namespace_1.x.add_mapping(item_2.namespace_2.y)


This code creates a mapping from `item_1.namespace_1.x` to `item_2.namespace_2.y`,
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

    # ...

    item1 = MyItem1()
    item2 = MyItem2()
    item2.t1.Q.add_mapping(item1.t1.T)


In this example, the Q parameter of MyItem2 is mapped to the T state of MyItem1.
This means that the value of item1.t1.T will be used as input (copied before any computation) for the parameter Q  in
the equations of MyItem2.


Mappings with assign and argumented assign
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mappings are an important aspect of the Numerous engine system, as they allow for the connection of variables and parameters between different items and subsystems. Mappings can be defined using two different types of assignments: assign and argumented assign.
    1. Assign Mapping: A basic mapping can be defined using the add_mapping() method. It takes in a variable or parameter as an argument, and assigns it as the output of the current variable or parameter. For example, if we have an item A with a variable x, and we want to assign the value of x to another item B's variable y, we can use the following code:
item_A.x.add_mapping(item_B.y)
This creates a mapping between the two variables, such that the value of x in item A is assigned to the value of y in item B.
    2. Argumented Assign Mapping: Another way to define mappings is by using the add_mapping_with_args() method, also known as argumented assign mapping. This method allows for additional arguments to be passed in, which can be used to define more complex mappings. For example, if we have an item A with a variable x, and we want to assign the value of x multiplied by a constant c to another item B's variable y, we can use the following code:
item_A.x.add_mapping_with_args(item_B.y, c=2)
This creates a mapping between the two variables, such that the value of x in item A multiplied by the constant c (2 in this case) is assigned to the value of y in item B.
It's important to note that mappings are only valid within the same namespace and they are only used during the simulation. They do not affect the model's state when it's not being solved.
The use of mappings allows for the creation of complex systems with a high degree of modularity, as different items and subsystems can be connected and reused easily. It's also a powerful tool for making the model more readable and maintainable.


Mappings with connector and ports:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Numerous engine provides a flexible way to model and simulate complex systems by using the concept of subsystems and connectors. A subsystem is a collection of items, each representing a part of the system, that are connected together through connectors.
Binding with Connector: In Numerous, a connector is a special type of item that is used to connect the inputs and outputs of two or more items. Connectors are used to define the relationships between items in a subsystem. A connector can be created by instantiating the Connector class from the numerous.engine.system.connector module. Once a connector is created, it can be used to bind the inputs and outputs of different items together. For example, to bind the input of item A to the output of item B, you can use the following code:

.. code::

    connector = Connector()
    itemA.input.add_mapping(connector.output)
    itemB.output.add_mapping(connector.input)

Mapping with Connector: In addition to binding, connectors can also be used to map the inputs and outputs of different items together. Mapping is similar to binding, but it allows for a more flexible way to connect items together. For example, instead of connecting the input of item A directly to the output of item B, you can use a connector to map the output of item B to a different input of item A. This can be useful when you want to connect multiple items together in a complex system.

.. code::
    connector = Connector()
    itemA.input1.add_mapping(connector.output)
    itemB.output.add_mapping(connector.input)

Ports in Subsystem: In Numerous, a subsystem is a collection of items that are organized and connected together to form a complete system. A subsystem can be created by instantiating the Subsystem class from the numerous.engine.system.subsystem module. Once a subsystem is created, it can be used to register items and connectors, and to define the relationships between them. One of the key features of a subsystem is the use of ports. Ports are used to define the inputs and outputs of the subsystem, and to connect the subsystem to other subsystems or to the external world.

.. code::

    subsystem = Subsystem()
    subsystem.register_items([item1, item2, connector])
    subsystem.register_input(connector.input, "input_port")
    subsystem.register_output(item2.output, "output_port")

In the example above, the input_port is defined as the input of the connector, and the output_port is defined as the output of item2. These ports can then be used to connect the subsystem to other subsystems or to the external world.






