Numerous Engine Model
==================
The Model class is initialized with a Subsystem object, which is the top-level container for the system's components. The Subsystem class allows users to organize the system's components into a hierarchical structure, with each Item representing a subsystem or a component of the system. Each Item can have one or more Namespace objects, each of which can contain a set of equations and variables.
The Model class provides the following key features:
    • Cloning: The Model class is designed to be clonable, which means that it can be used to create multiple independent simulations of the same system. Cloning is useful when you need to run multiple simulations with different initial conditions or parameters. The clone() method creates a new Model object with the same Subsystem hierarchy, but with different states and parameters.
    • Caching: The Model class is designed to use caching to improve performance. This means that the results of expensive calculations are stored in memory and reused when the same calculation is needed again. The Model class uses caching to improve the performance of the equation evaluation and Jacobian calculation. The caching is automatically enabled, but it can be disabled by passing the caching=False parameter to the Model constructor.
    • State and Time Events: The Model class allows users to define and handle events that occur at specific times or states of the system. Events can be used to change the behavior of the system at specific times or states, such as when a component reaches a critical temperature or when a certain condition is met. Events can be defined by decorating a method with the Event decorator and passing the time or state at which the event should occur as an argument. The method will be called when the event occurs, and it can modify the system's states or parameters as needed.
    • Global Variables : The Model class allows users to add global variables to the model, which can be accessed by all items and subsystems of the system. This can be useful when a value needs to be shared across the entire system. Global variables can be added to the model by passing a list of tuples, where each tuple contains the name of the variable and its value to the Model constructor.
    • LLVM support: The Model class can use the LLVM library to improve the performance of equation evaluation. This is useful when the system has a large number of equations or when the equations are computationally expensive. LLVM support can be enabled by passing the use_llvm=True parameter to the Model constructor.
Overall the numerous engine is a powerful tool that can help in simulating complex dynamic systems, it's well suited for systems engineering, control systems, and process engineering. It's well tested, easy to use and it offers the possibility to optimize performance by caching and LLVM support.

Model Logger levels
^^^^^^^^^^^^^^^^^^

Logging is used to track the progress and performance of simulations and to provide information about the state of the system at any point in time. The engine uses the Python logging module to handle logging, which allows users to control the level of logging and to redirect log messages to different outputs.
The engine provides several levels of logging, which are organized in a hierarchical manner. The levels are:
-DEBUG: Detailed information, typically of interest only when diagnosing problems.
-INFO: Confirmation that things are working as expected.
-WARNING: An indication that something unexpected happened or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.
-ERROR: Due to a more serious problem, the software has not been able to perform some function.
-CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
The logging level can be set globally for the entire engine using the numerous.engine.logging.set_level() function, or on a per-Model basis using the model.logger.setLevel() method. By default, the logging level is set to WARNING.
In addition to setting the logging level, it is also possible to redirect log messages to different outputs using the logging.basicConfig() function. For example, log messages can be sent to a file or to the console.
Here is an example of how to set the logging level to DEBUG and redirect log messages to a file:
.. code::
    import logging
    import numerous

    numerous.engine.logging.set_level(logging.DEBUG)
    logging.basicConfig(filename='numerous_debug.log', level=logging.DEBUG)
You can also configure the logging for certain components of the model by providing the package name in the basicConfig method.
.. code::
    logging.basicConfig(filename='model_debug.log', level=logging.DEBUG, format='%(levelname)s:%(name)s: %(message)s', )
It's important to note that increasing the logging level might make the simulation slower, that's why it's good practice to use the minimum level of log to debug the problem.
In addition to the built-in logging functionality, the Numerous engine also provides a Historian class that can be used to track the values of variables over time. This can be useful for analyzing the behavior of a system during a simulation, and for plotting and visualizing the results.

Model state and time events
^^^^^^^^^^^^^^^^^^
Events can be used to change the values of parameters or to perform other actions, such as updating the model's state or running additional calculations.
State events are triggered when the value of a state variable reaches a specific threshold. The value of the state variable is checked at each time step during the simulation, and if it crosses the threshold, the event is triggered. For example, if a state variable represents the position of a moving object, a state event could be used to detect when the object reaches a specific point in space.
Time events are triggered at a specific point in time, regardless of the value of any state variables. For example, a time event could be used to update the value of a parameter at a specific time, or to run additional calculations at a specific point in the simulation.
To add a state event to a model, you can use the add_state_event() method on the Model class. This method takes in the following arguments:
    • name: a string that identifies the event.
    • state: the state variable that the event is associated with.
    • threshold: the value of the state variable that the event is triggered at.
    • event_function: the function that is called when the event is triggered. This function takes in the current time and the current state of the model as arguments.
    • direction: the direction of the state variable crossing the threshold. this could be "both", "rising" or "falling"
For example, the following code creates a state event that is triggered when the value of the x state variable reaches 10:
model = Model(system)
model.add_state_event("x_event", state = system.x, threshold = 10, event_function = some_function, direction = "rising")
To add a time event, you can use the add_time_event() method on the Model class. This method takes in the following arguments:
    • name: a string that identifies the event.
    • time: the time at which the event is triggered.
    • event_function: the function that is called when the event is triggered. This function takes in the current time and the current state of the model as arguments.
For example, the following code creates a time event that is triggered at time t = 5:
model = Model(system)
model.add_time_event("t_event", time = 5, event_function = some_function)
It should be noted that, events functions can only contain simple mathematical operations and assignments, while they cannot contain any logic operations like if else or loops.
It is also important to note that when using events, the order in which the events are defined may affect the simulation results.

Model cloning and caching
^^^^^^^^^^^^^^^^^^

Cloning a model allows you to create a copy of an existing model, which can then be used for further simulations or analysis without modifying the original model. This can be useful when you want to run multiple simulations with slightly different parameter values or initial conditions. To clone a model, you can call the clone() method on an existing model object, and then pass the resulting object to a new simulation.
Caching a model allows you to save the results of a simulation to disk, so that they can be reused later without having to re-run the simulation. This can be useful when you have a large or computationally expensive model and you want to avoid running the simulation multiple times. To cache a model, you can call the cache() method on a simulation object, and then pass the resulting object to a new simulation.
The clone() and cache() methods are both optional arguments of Model class and can be passed as True or False during instantiation of the Model. By default, both clone and cache arguments are set to False, and the model is not clonable or cacheable.
When caching is enabled, the states of the model will be saved to the cache file before the simulation starts. And when the simulation is run again, the model will load the states from the cache file and continue the simulation from there.
When using cloning and caching together, it is important to note that cloning a cached model will also clone the cache file. This means that if you make changes to the cloned model, the original model and its cache file will not be affected.
It is important to note that caching and cloning of the model will increase the memory usage of the engine and thus it should be used with care, especially when dealing with large models.
Example:
.. code::
    #creating a model
    system = S2N("S2", 2)
    model = Model(system)

    #cloning the model
    cloned_model = model.clone()

    #caching the model
    simulation = Simulation(model, t_start=0, t_stop=2, num=2)
    simulation.cache()

    #cloning the cached model
    cloned_cached_model = simulation.model.clone()
As a best practice, it is recommended to use caching and cloning as needed, and avoid using them when not necessary. This can help to optimize the performance and memory usage of the engine.





Model external mappings and global variables
^^^^^^^^^^^^^^^^^^
 External mappings allow variables in one part of the system to be connected to variables in another part of the system, allowing the system to be more modular and easier to understand.
In the Numerous Engine, external mappings are created by calling the add_mapping() method on a variable. This method takes a single argument, which is the variable that the current variable is being mapped to. For example, to map a variable x to a variable y, you would call x.add_mapping(y). This creates a mapping between x and y, and any changes to the value of x will be reflected in the value of y.
External mappings can be created at both the system and model level. On the system level, external mappings are used to connect the inputs and outputs of different subsystems. For example, the output of one subsystem could be mapped to the input of another subsystem. This allows the subsystems to be connected together to form a larger system.
On the model level, external mappings are used to connect variables within a subsystem. For example, one variable in a subsystem could be mapped to another variable in the same subsystem. This allows the subsystem to be divided into smaller, more manageable parts.
One key aspect of external mappings is that they are not bidirectional, if you want to change the value of y, you have to change the value of x.
It is also important to note that external mappings can only be created between variables of the same type, such as two states or two parameters. Attempting to create a mapping between a state and a parameter will result in an error.
When creating external mappings, it's important to keep in mind that they can only be created between variables that are part of the same model. In other words, you cannot create an external mapping between a variable in one model and a variable in another model.
In order to use external mappings in the Numerous Engine, you will need to create an instance of the Model class, and register the items and subsystems that you want to use in the system. Once the model is created, you can create external mappings between the variables in the system by calling the add_mapping() method on the variables.
Here is an example of how to use external mappings in the Numerous Engine:
.. code::
    from numerous.engine.model import Model
    from numerous.engine.system import Item, Subsystem

    class MyItem(Item):
        def __init__(self, tag):
            super().__init__(tag)
            self.create_namespace("my_ns")
            self.my_ns.add_state("x", 0)
            self.my_ns.add_state("y", 0)

    class MySubsystem(Subsystem):
        def __init__(self, tag):
            super().__init__(tag)
            item1 = MyItem("item1")
            item2 = MyItem("item2")
            item1.my_ns.x.add_mapping(item2.my_ns.y)
            self.register_items([item1, item2])

    # Create the model and register the subsystem
    model = Model(MySubsystem("my_subsystem"))
In this example, we create two instances



The Numerous engine allows the use of global variables in equations, which can be added to the model by passing a list of global variables in the form of tuples with variable name and value to the global_variables parameter of the Model class.
For example, to add a global variable g with a value of 9.81 to the model, you would create the model as follows:
model = Model(system, global_variables=[("g", 9.81)])
Once a global variable is added to the model, it can be accessed within the equation functions by using the scope object passed to the equation function, for example:
.. code::
    @Equation()
    def eval_(self, scope):
        acceleration = scope.g * scope.mass
It's also possible to add global variables on system level by using the add_global_variable method of the Subsystem class, this will add the variable to the system and all sub-systems and items.
Additionally, you can also add global variables on item level by using the add_global_variable method of the Item class, this will add the variable to the item and its namespaces.
It's important to note that global variables are constant, meaning that their value will not change during the simulation, but you can change the value by passing a new value to the global_variables parameter of the Model class on the next simulation.
In addition, it is also possible to access global variables on external mappings, by referencing the variable with the g prefix and the variable name, for example:
.. code::
    item.t1.T_o.add_mapping(item2.t2.T, global_mapping={"g.g": "g"})
This will map the g.g variable of item1 to g variable of item2.
Using global variables can help in situations where you want to use the same value in multiple equations, or when you want to change the value of a parameter that is used in multiple equations without modifying each equation individually.




