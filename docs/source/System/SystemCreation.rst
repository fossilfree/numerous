Creating a System
==================

To enable building complex systems in a modular way, subsystems are used to define
combinations of items and subsystems. Items and subsystems can be registered to a subsystem
to denote it as the parent for the registered items.
This allows for a hierarchical representation of the model system.

Starting with Connector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we will look at how to create a simple system that contains 2 basic items and the connector.
We will start by inheriting a :class:`numerous.engine.system.ConnectorTwoWay`
that is special case of :class:`numerous.engine.system.Connector`
with two sides with default names for them side1 and side2.

.. code::

    class ThermalConductor(ConnectorTwoWay):
        def __init__(self, tag):
            super(ThermalConductor, self).__init__(tag)


Alternatively we can specify our own names for the sides of the

.. code::

    super().__init__(tag,side1_name='inlet', side2_name='outlet')


After we are creating  namespace to this item and adding two equations to it.
`update_bindings` flag show that we expect to have variables of HeatConductance in side1 and side2.
:class:`numerous.engine.OverloadAction` enum is describing an action that should be used in case
of multiple reassign to variable during same step. `OverloadAction.SUM` will sum  values instead of overwriting.

.. code::

        hc1 = HeatConductance(h=1001)
        hc2 = HeatConductance(h=1001)

        thermal = self.create_namespace('thermal')

        thermal.add_equations([hc1, hc2], on_assign_overload=OverloadAction.SUM, update_bindings=True)

Now we have namespace created not only inside the item but inside the defined bindings
(in this case inside side1 and side2).
We can continue with mapping variables inside the namespace to variables in bindings.
Inside an item, mappings are used to map the value of one variable onto another.
This is used to tell the engine that the value of one variable inside one equation
is actually the value of another variable inside another equation.
Mappings are defining interactions between variables not in the same namespace explicitly.



.. code::

        thermal.T1 = self.side1.thermal.T
        self.side2.ns.thermal = thermal.T1


Now in equation in namespace thermal any access  to value of variable
T1 will be readdressed to item that is binded to side1.

Creating a System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now we can create a class that inherits :class:`numerous.engine.system.Subsystem`.
and inside of it we define 2 basic items and 1 Connector item.

.. code::

    class System_Level_1(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            TM1 = ThermalMass('TM1')
            TM2 = ThermalMass('TM2')
            C12 = ThermalConductor('TC1')

            #now we have to registered created items for the current subsystem
            self.register_items([TM1, TM2, C12])


Different objects in the system needs a way to interact with each other.
This can be achieved by passing some object instances
as an argument to another item and mapping there variables explicitly.
However, in numerous bindings can be used to specify
prototypes of other items and their required namespaces and variables.

Bindings furthermore enables two items to bi-directionally interact with each other,
since both items can be instantiated and then bounded to each other.

Having an instance of  ConnectorTwoWay and two items we can create binding by defining corespondance between
sides and items:

.. code::

            C12.bind(side1=TM1,side2=TM2)



If we are planning to use some parts of the subsystem for other binding we can define ports inside an subsystem:

.. code::

            self.add_port('inlet', TM1)
            self.add_port('outlet', TM2)



