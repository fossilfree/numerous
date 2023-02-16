Events
==================



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