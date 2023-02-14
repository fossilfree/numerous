System
==================

To enable building complex systems in a modular way, subsystems are used to define
combinations of items and subsystems. Items and subsystems can be registered to a subsystem
to denote it as the parent for the registered items.
This allows for a hierarchical representation of the model system.
and connect Items with the equations and variables that make up a system
A System object is made up of one or more Item objects, each of which represents a single component of the system.
One of the key features of the Numerous engine is the ability to map variables between different ``Item`` objects.

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


Creation and working with systems that include fmu subsystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FMUs (Functional Mock-up Units) can be used to import the system from other modeling languages that support the FMU standard.
FMUs can be integrated into a larger system modeled using the Numerous engine by creating an FMUSubsystem object and
registering it as a child of a Subsystem object or be simulated on is own.
To create an FMUSubsystem, you need to provide the path to the FMU file,
and the name of the model and the output variable(s) of the FMU that you want to use.
The FMUSubsystem object can then be added to the system using the register_items() method of the parent Subsystem object.
For example, let's say you have an FMU file called 'my_fmu.fmu' that models a mechanical system, and you want to use the
output variable 'displacement' from the model 'MyModel'. You can create an FMUSubsystem object and add it to a system as follows:
from numerous.engine.system import Subsystem, FMUSubsystem

.. code::

    # Create a Subsystem object to represent the overall system
    system = Subsystem("my_system")

    # Create an FMUSubsystem object for the mechanical system
    fmu_subsystem = FMUSubsystem("mechanical_system", "my_fmu.fmu", "MyModel", ["displacement"])

    # Register the FMU subsystem as a child of the overall system
    system.register_items(fmu_subsystem)

In addition, you can create mappings between variables in the FMU subsystem and variables in the rest of the system, allowing the FMU to interact with other parts of the system.
For example, the following code snippet shows how to create a mapping between the input variable 'force' in the FMU and the output variable 'F_out' in the parent subsystem:
fmu_subsystem.fmu_inputs.force.add_mapping(system.F_out)
Once the FMU subsystem is added to the system, it can be simulated along with the other parts of the system using the Simulation class, just like any other item in the system. The Simulation class will automatically take care of initializing and communicating with the FMU during the simulation.
Keep in mind FMU are independent models and their time step is independent from the time step of the system. Also, it's better to use the same solver in the FMU and in the system to ensure consistency.
Note that, it's also possible to use the FMU in stand alone mode, if you want to use the FMU outside of the system and use it as a black box.
It's also worth noting that some FMUs may have additional requirements, such as external libraries or specific versions of Python or other dependencies. Be sure to check the documentation for the FMU you are using to ensure that you have the necessary dependencies installed.




State and time Events on system level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Numerous engine, a state event is a condition that is checked at each time step of a simulation to determine if a specific action should be taken. State events can be used to change the value of a state variable or parameter, or to change the integration method of the solver. State events are defined on a per-system basis, and are added to a system using the add_state_event() method.
A state event is defined by a condition, which is a mathematical expression that is evaluated at each time step. If the condition is true, the action specified in the event is executed. The condition can be a simple comparison, such as x > 5, or a more complex expression involving multiple state variables and parameters.
The action of a state event can be one of the following:
    • Change the value of a state variable or parameter.
    • Change the integration method of the solver.
    • Execute a custom function that can perform any other action.
For example, consider a system with a state variable x and a parameter p. The following code defines a state event that changes the value of x to 10 when x becomes greater than 5 and changes the value of p to 3:

.. code::

    class MySystem(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            self.add_state("x", 0)
            self.add_parameter("p", 1)
            self.add_state_event("x > 5", action="x = 10; p = 3")

Similarly, Time events are a way of specifying conditions that are checked at specific times, rather than at each time step. They can be used, for example, to change the value of a state variable or parameter at a specific time, or to change the integration method of the solver at a specific time. They are defined on a per-system basis, and are added to a system using the add_time_event() method.
A time event is defined by a time and a condition, which is a mathematical expression that is evaluated at the specified time. If the condition is true, the action specified in the event is executed. The condition can be a simple comparison, such as x > 5, or a more complex expression involving multiple state variables and parameters.

.. code::

    class MySystem(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            self.add_state("x", 0)
            self.add_parameter("p", 1)
            self.add_time_event(2, "x > 5", action="x = 10; p = 3")

It's important to note that state and time events are executed before the update of the state variables and parameters, so they can also be used to change the state of the system before the next step of the simulation.
Also, when using state and time events, the model needs to be solved using the solve_with_events() method, to execute events at the appropriate times.




Registering of special methods on  on subsystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Numerous engine allows users to register additional methods on subsystems and items to be run at specific points during the simulation. These methods can be used to perform custom computations or update the state of the system.
1. Run after solve method registration: The run_after_solve method is a function that is called after the system of equations is solved for each time step. It can be used to perform custom computations on the state variables of the system, such as calculating additional quantities or updating the state of the system based on the results of the simulation. To register a run_after_solve method on a subsystem or item, call the register_run_after_solve method on the subsystem or item and pass in the method as an argument. For example:

.. code::

    def my_run_after_solve(self, scope):
        scope.x = scope.x + 1
    subsystem.register_run_after_solve(my_run_after_solve)
2. Post step method registration: The post_step method is a function that is called after the run_after_solve method is called, and it can be used to perform additional computations or update the state of the system based on the results of the simulation. To register a post_step method on a subsystem or item, call the register_post_step method on the subsystem or item and pass in the method as an argument. For example:

.. code::

    def my_post_step(self, scope):
        scope.x = scope.x + 1
    subsystem.register_post_step(my_post_step)

In summary, the Numerous engine provides several mechanisms for creating and managing
connections between subsystems and items, including ports, connectors, and mapping
which allows the user to effectively simulate complex systems of equations.

Set variables and Item set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Numerous engine, an Item represents a single component of a system, and a Subsystem represents a collection of multiple Item objects that work together to form a larger system. When creating a Subsystem, it's possible to register a list of Item objects as a set, using the register_items method.
The register_items method accepts a list of Item objects, and an optional structure argument that defaults to ItemsStructure.SEQUENCE. By passing ItemsStructure.SET as the value of the structure argument, the registered Item objects will be treated as a set, rather than a sequence. This can be useful when working with systems where the order of the items doesn't matter and only unique items are considered.
Here is an example of how to create a Subsystem and register a list of Item objects as a set:
from numerous.engine.system import Subsystem, Item, ItemsStructure

.. code::

    class MyItem(Item):
        def __init__(self, tag):
            super().__init__(tag)

    class MySubsystem(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            items = [MyItem("item1"), MyItem("item2"), MyItem("item3")]
            self.register_items(items, structure=ItemsStructure.SET)

In this example, we create a MySubsystem class that inherits from Subsystem and a MyItem class that inherits from Item. We then create a list of MyItem objects and pass it to the register_items method, along with the structure argument set to ItemsStructure.SET.
By registering the items as a set, it allows us to make sure that the subsystem only contains unique items and also allows us to use set operations like union and difference on items list.
It's important to note that, when using the ItemsStructure.SET, items passed to the register_items method must have unique tags. If there are duplicates, it will raise an error.