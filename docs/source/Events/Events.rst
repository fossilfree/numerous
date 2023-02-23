Events
==================

In the Numerous engine, a ``state event``  is a condition that is evaluated at each ``timestep``  of a simulation.
If the condition is met, a specified action is triggered. ``State events``  can be used to change the value of
a state variable or parameter, or to change the integration method of the solver.

``State events``  are defined by a mathematical expression or a condition that is evaluated after each solver convergence.
This condition is called the ``zero-crossing`` condition, which refers to the point at which a mathematical expression
changes sign. If the condition is true, the action specified in the event is executed. The action of a ``state event``
can include for example changing the value of a state variable or parameter

A ``time event``  is a type of event that is conditioned on time and is only executed during specific
time steps. The action of a ``time event``  is similar to a ``state event`` , where the specified action is triggered if
the time condition is met. However, unlike a state event, the ``time event``
is triggered only at specific times during the simulation.
We use ``variable path`` to address variables in dictionaries that are passed to functions.


Adding State and Time events
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We can add ``state`` and ``time event`` for both ``model`` and ``system``
using ``add_event`` and ``add_timestamp_event`` functions.

Here's how you can add ``state event``:

Adding ``state event``:

    #. Create a ``condition`` function that checks for the desired state. This function should take the current time and ``system`` variables as input.
    #. Create a ``action`` function that will be called when the state event is triggered. This function should take the current time and ``system`` variables as input, and can modify the variables as needed.
    #. Call the ``add_event`` method on the system object, passing the name of the event, the state-checking function, and the state-modifying function as arguments. It is also possible to specify direction of ``zero crossing``.

Here's an example code snippet that adds a state event to a system:

.. code::


    # returns position to find zero crossing using root finding algorithm of scipy solver
    def hitground_event_fun(t, states):
            return states['t1.x']

    # change direction of movement upon event detection and reduce velocity
    def hitground_event_callback_fun(t, variables):
            velocity = variables['t1.v']
            velocity = -velocity * (1 - variables['t1.f_loss'])
            variables['t1.v'] = velocity
            variables['t1.t_hit'] = t

    system.add_event("hitground_event", hitground_event_fun, hitground_event_callback_fun)

To add ``time event`` function  `` add_timestamp_event``  is used. We follow the same steps,
but instead of ``action`` function we specify ``timestep`` array or ``periodicity`` constant.

.. note::

    ``Variable path`` used in condition and action function is relative to the system we add event to. If we add event to the model we have to use path to the ``root system``.

