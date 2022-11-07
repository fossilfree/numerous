Creating an Item
==================
The basic building block of models in numerous. Any basic object (
like a pump or pipe for instance) is created as a class extending the :class:`numerous.engine.system.Item`.
Item defines an interface for the numerous engine to determine how different
objects are interacting and from that setup the function that solves the system of objects.

Creating an empty Item
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To create an empty item we inherit :class:`numerous.engine.system.Item` and pass a tag parameter to it's constructor.

.. code::

    class ThermalMass(Item):
        def __init__(self, tag):
            super().__init__(tag)

After defining a ThermalMass  we can instantiate an object and work with it in numerous engine.

.. code::

    tm = ThermalMass(tag='ThermalMass1')

Adding a namespace to an empty Item
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order avoid unintended variable manipulation from
different equations with same variable names :class:`numerous.engine.system.Namespace` are created.
Equations added to the same namespace will operate on the same variables
– equations added to different namespaces will operate on different
sets of variables and interactions will have to be explicitly defined through mappings.
Namespaces can be added inside Item constructor.

.. code::

    class ThermalMass(Item):
        def __init__(self, tag='ThermalMass'):
            super().__init__(tag)
            mechanics = self.create_namespace('mechanics')

Now we can add :class:`numerous.engine.system.Variable` and :class:`numerous.engine.system.Equation` to the namespaces.
On adding an Equation all variables required for the equation will be created.