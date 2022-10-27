Working with the model
======================
For the engine to setup the equations,
variables and mappings between these the top-level subsystem is passed to a model object.
:class:`numerous.engine.model.Model`  traverses the system to
collect all information needed to pass to the solver for computation –
the model also back-propagates the numerical results from the solver into the system,
so they can be accessed as variable values there.
Model is created from the :class:`numerous.engine.system.Subsystem`:

.. code::

     m1 = Model(model_system)
     sim = Simulation(m1, t_start=0, t_stop=10, num=100, max_step=0.01)


Adding callbacks to model
^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible to add two types of callbacks that will be executed during simulation.
- `callback` functions. These are functions that are called each time a solver step has been completed.



.. code::

     m1 = Model(model_system)
     m1.add_callback('hitground', hitground_callback_ms1)
     sim = Simulation(m1, t_start=0, t_stop=10, num=100, max_step=0.01)

- `event` functions. An event function uses a root-finding algorithm to detect when a certain condition is triggered.
    A callback can be attached to run after any specific event.

.. code::

     m1 = Model(model_system)
     m1.add_event("hitground_event", hitground_event_fun)
     m1.add_event_callback("hitground_event", hitground_event_callback_fun)
     sim = Simulation(m1, t_start=0, t_stop=10, num=100, max_step=0.01)



Creating aliases for variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Having many nested subsytems can make it difficult to follow the changes of important variable.
to highlight one of the variables we can add a special alias to it. Later we can only save such variables to history
data frame


Saving and restoring state of the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
It is possible to periodically save the states of the system to the file. Such
that the long running solution would not be lost.

.. code::

    hdf = HistoryDataFrame()
    m1 = Model(S3('S3'), historian=hdf)

    c1 = _SimulationCallback("test")
    m1.save_variables_schedule(0.1, filename)

    s1 = Simulation(m1, t_start=0, t_stop=2, num=100)

    hdf2 = HistoryDataFrame.load(filename)
    m2 = Model(S3('S3'), historian=hdf2)
    m2.restore_state()