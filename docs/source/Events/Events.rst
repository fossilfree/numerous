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

State and time Events on system level
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When working with systems, we can add both state and time events.
Here's how you can add State Events on the system level:

Adding State Events:

    #. Create a condition function that checks for the desired state. This function should take the current time and system variables as input.
    #. Create a action function that will be called when the state event is triggered. This function should take the current time and system variables as input, and can modify the variables as needed.
    #. Call the add_event method on the system object, passing the name of the event, the state-checking function, and the state-modifying function as arguments. It is also possible to specify direction of zero crossing.

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

Model state and time events
^^^^^^^^^^^^^^^^^^

.. code::

    def timestamp_callback(t, variables):
        print(t)


    @pytest.mark.parametrize("use_llvm", [True, False])
    def test_bouncing_ball(use_llvm):
        model_system_2 = ms1(Ball(tag="ball", g=9.81, f_loss=0.05))
        m1 = Model(model_system_2, use_llvm=use_llvm)

        m1.add_event("hitground_event", hitground_event_fun, hitground_event_callback_fun)

        sim = Simulation(m1, t_start=0, t_stop=tmax, num=num)

        sim.solve()
        asign = np.sign(np.array(m1.historian_df['S1.ball.t1.v']))
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        args = np.argwhere(signchange > 0)[2:].flatten()
        assert approx(m1.historian_df['time'][args[0::2][:5]], rel=0.01) == t_hits[:5]


    @pytest.mark.parametrize("use_llvm", [True, False])
    def test_with_full_condition(use_llvm):
        model_system_2 = ms1(Ball(tag="ball", g=9.81, f_loss=0.05))
        m1 = Model(model_system_2, use_llvm=use_llvm)

        m1.add_event("hitground_event", hitground_event_fun_g, hitground_event_callback_fun)
        m1.add_timestamp_event("timestamp_event", timestamp_callback, timestamps=[0.11, 0.33])