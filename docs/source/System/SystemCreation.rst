`System
==================

To enable building complex systems in a modular way, subsystems are used to define
combinations of items and subsystems. Items and subsystems can be registered to a subsystem
to denote it as the parent for the registered items.
This allows for a hierarchical representation of the model system
and connect Items with the equations and variables that make up a system.

Creating a System
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Creating a system in the Numerous modeling and simulation engine involves defining a hierarchy of subsystems,
which are collections of interconnected components. Here are the general steps:

#. Create a class that inherits from the ``Subsystem`` class, which is a base class for all subsystems in Numerous.
#. Define the subsystem's properties, which are represented by classes that inherit from ``Item``. These can include ``namespaces`` and ``mappings``.
#. Register items and subsystems of the subsystem by calling the ``register_items`` method on the subsystem instance.
#. Instantiate the system by creating an instance of the  ``Subsystem``.

Here's an example of how to create a simple system with two thermal masses:

.. code::

    class ThermalSystem(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            TM1 = ThermalMass('TM1')
            TM2 = ThermalMass('TM2')
            self.register_items([TM1, TM2])



Creation and working with systems that include fmu subsystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``FMUs`` (Functional Mock-up Units) can be used to import the system from other modeling
languages that support the ``FMU`` standard.
FMUs can be integrated into a larger system modeled using the Numerous engine by creating an ``FMUSubsystem`` object and
registering it as a child of a ``Subsystem`` object or be simulated on is own.
To create an FMUSubsystem, you need to provide the path to the FMU file,
and the name of the model and the output variable(s) of the FMU that you want to use.
The FMUSubsystem object can then be added to the system using the ``register_items()`` method of the parent Subsystem object.
For example, let's say you have an FMU file called 'my_fmu.fmu' that models a mechanical system, and you want to use the
output variable 'displacement' from the model 'MyModel'. You can create an FMUSubsystem object and add it to a system as follows:
from numerous.engine.system import Subsystem, FMUSubsystem

.. code::

    # Create a Subsystem object to represent the overall system
    system = Subsystem("my_system")

    # Create an FMUSubsystem object for the mechanical system
    fmu_subsystem = FMUSubsystem("mechanical_system.fmu", "mechanical_system")

    # Register the FMU subsystem as a child of the overall system
    system.register_items([fmu_subsystem])

In addition, you can create mappings between variables in the ``FMUSubsystem`` and variables in the rest of the system,
allowing the ``FMU`` to interact with other parts of the system.
Once the ``FMUSubsystem`` is added to the system, it can be simulated along with the other parts of the system
using the ``Simulation`` class, just like any other item in the system. The ``FMUSubsystem`` will automatically take
care of initializing and communicating with the FMU during the simulation.
Note that, it's also possible to use the FMU in stand alone mode, if you want to use the FMU outside of the system and
use it as a black box.

.. note::

    Some FMUs may have additional requirements,such as external libraries or specific versions of operating system
or other dependencies.Be sure to check the documentation for the FMU you are using to ensure that you have
the necessary dependencies installed.





Registering of special methods on  on subsystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can register additional methods on subsystems and items to be run at specific points
during the simulation without any conditions.
These methods can be used to perform custom computations, update or save the state of the system.

Run after solve method
----------------

The ``run_after_solve`` is a field in ``Subsystem`` class that  contains a list of names of methods that will be
called after the system of equations is solved for each time step. Methods should be part of  our system class instance.
 To register a ``run_after_solve`` method on a subsystem or item, call the register_run_after_solve
method on the subsystem or item and pass in the method as an argument. For example:

.. code::

class Test_Subsystem(Subsystem):
    def __init__(self tag: str):
        super().__init__(tag)
        external_id = ""
        self.post_step = ['_terminate']

        def _terminate():
            print(external_id)

        self.run_after_solve = _terminate


Post step method
----------------

The ``post_step`` is a field in ``Subsystem`` class that  contains a list of names of methods that will be
called after  each solver convergence. Methods should be part of  our system class instance. For example:

.. code::

class Test_Subsystem(Subsystem):
    def __init__(self tag: str):
        super().__init__(tag)
        external_id = ""
        self.post_step = ['_execute']

        def _execute():
            print(external_id)

        self.fmi2Terminate_ = _execute



Set variables and Item set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When creating a ``Subsystem``, it's possible to register a list of ``Item`` objects as a set,
using the ``register_items`` method.
The ``register_items`` method accepts a list of ``Item`` objects, and an optional structure argument that defaults to
``ItemsStructure.LIST``. By passing ``ItemsStructure.SET`` as the value of the ``structure`` argument, the registered ``Item``
objects will be treated as a set, rather than a list. This means it is expected that all items in the set are of the
same type and dont have mapping that define order of computation between them. By using ``ItemsStructure.SET``
we speed up computation of a similar not interconnected ``Items``.


.. code::
    class MyItem(Item):
        def __init__(self, tag):
            super().__init__(tag)

    class MySubsystem(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            items = [MyItem("item1"), MyItem("item2"), MyItem("item3")]
            self.register_items(items, structure=ItemsStructure.SET)