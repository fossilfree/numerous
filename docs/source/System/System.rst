
Numerous Engine System
==================

At the front of the numerous engine is the :class:`numerous.engine.system.Subsystem` class, which is used to define and connect  Items with the equations and variables that make up a system.
A System object is made up of one or more Item objects, each of which represents a single component of the system. Each Item object can contain one or more namespaces, which are used to organize the equations and variables for that component. Each namespace represents a specific physical domain or area of the system, and can contain its own set of equations and variables.
namespace  can contain several types of variables: state, parameter, and constant. State variables are quantities that change over time by the solver, such as the position or velocity of an object. Parameters are quantities that can change over time but doesn't have a derivative. Constants are fixed quantities that do not change over time, such as the mass or length of an object.
Each namespace can contain one or more Equation objects, which are used to define the mathematical relationships between the variables in the namespace. These equations are written as methods on a class that inherits EquationBase class and decorated with the Equation decorator.
One of the key features of the Numerous engine is the ability to map variables between different Item objects. This allows you to define the relationships between different components of the system and to simulate the system as a whole.
To create a mapping between two variables, you can use the add_mapping method on the variable you want to map from. For example:

.. code::
    item_1.namespace_1.x.add_mapping(item_2.namespace_2.y)

This creates a mapping from `item_1.namespace_1.x` to `item_2.namespace_2.y`, so that the value of x is used as the value of y in the simulation.
Once the mappings have been defined, you can create a Model object from the System object and use it to run a simulation.

Namespaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Namespaces are created by calling the create_namespace() method on an Item or Subsystem object. For example:
.. code::
    class MyItem(Item):
        def __init__(self, tag='my_item'):
            super().__init__(tag)
            # Create a namespace for our equations
            mechanics = self.create_namespace('mechanics')
            # Add variables to the namespace
            mechanics.add_state('x', 0)
In the above example, a namespace called mechanics is created on the MyItem object, and a state variable x is added to it.
Once a namespace is created, equations can be added to it by calling the add_equations() method on the namespace and passing in a list of equation objects as an argument.
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

In this example, MyItem class is added to mechanics namespace.

Mappings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mapping is used to connect variables between different Item and Subsystem objects within a system. This allows for the variables of one object to be used as input for the equations of another object. For example, the temperature of one object can be used as input for the heat transfer equation of another object.
Mapping is done by calling the add_mapping() method on a variable, and passing in the variable that it should be mapped to. For example:
.. code::
    class MyItem1(Item):
        def __init__(self, tag='item1'):
            super().__init__(tag)
            t1 = self.create_namespace('t1')
            t1.add_state('T', 0)

    class MyItem2(Item):
        def __init__(self, tag='item2'):
            super().__init__(tag)
            t1 = self.create_namespace('t1')
            t1.add_parameter('Q', 0)
            t1.add_equation(MyEquation())

    # ...

    item1 = MyItem1()
    item2 = MyItem2()
    item2.t1.Q.add_mapping(item1.t1.T)
In this example, the Q parameter of MyItem2 is mapped to the T state of MyItem1. This means that the value of item1.t1.T will be used as input for the Q parameter in the equations of MyItem2.
It is important to note that if a variable is mapped to another variable, it will take on the same value




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





Creation and working with systems that include fmu subsystem
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the Numerous engine, FMUs (Functional Mock-up Units) can be used to simulate the behavior of subsystems modeled using Modelica or other modeling languages that support the FMU standard. FMUs can be integrated into a larger system modeled using the Numerous engine by creating an FMUSubsystem object and registering it as a child of a Subsystem object.
To create an FMUSubsystem, you need to provide the path to the FMU file, and the name of the model and the output variable(s) of the FMU that you want to use. The FMUSubsystem object can then be added to the system using the register_items() method of the parent Subsystem object.
For example, let's say you have an FMU file called 'my_fmu.fmu' that models a mechanical system, and you want to use the output variable 'displacement' from the model 'MyModel'. You can create an FMUSubsystem object and add it to a system as follows:
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

