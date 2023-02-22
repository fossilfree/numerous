Model
==================

The ``Model`` class is initialized with a Subsystem object, which is the top-level container for the system's components.
The ``Subsystem`` class allows users to organize the system's components into a hierarchical structure,
with each ``Item`` representing a subsystem or a component of the system.


Model creation
^^^^^^^^^^^^^^^^^^

To create a ``model`` in Numerous Engine, we need to define the ``subsystems`` which are a set of subsystem and Item classes
that define the components of the model. All subsystems should be registered in one ``subsystem``.

After defining the ``subsystems``, we can initialize the model and create an instance of the model class with
the main  ``subsystem`` as a parameter. The model is automatically compiled during initialization, which can take some time.

The initialized model object contains compiled functions for the solver. It is possible to add ``events`` to
the model after compilation, as events are  compiled during the initialization of the ``solver``.


Model external mappings and global variables
^^^^^^^^^^^^^^^^^^



Model Logger levels
^^^^^^^^^^^^^^^^^^



Model cloning and serialization
^^^^^^^^^^^^^^^^^^

Cloning a ``model`` allows you to create a copy of an existing ``model``. if model have to be cloneable
This can be useful when you want to run multiple simulations with slightly different parameter
values or initial conditions. To clone a ``model``, you can call the ``clone()``
method on an existing ``model`` object, and then pass the resulting object to a new simulation.

Exporting a ``model`` allows you to save the compiled ``model`` to disk, so that it can be reused later
without having to re-run the compilation process.  To export a ``model,`` you can
can provided with export_model=True argument during initialization. Model will be exported to he path specified in
``EXPORT_MODEL_PATH``  environment variable.
To restore exported  model we have to use ``from_file(filename: str)``  method, it will return a model ready
to simulation.



Getting results of the computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get results after the simulation is finished we need to read ``historian_df``.
``historian_df`` is an attribute of a Numerous model that stores a pandas ``DataFrame`` of the simulation history.

``historian_df``  attribute  is generated on a ``model`` instance after the simulation has completed .
The ``DataFrame`` contains a row for each time step of the simulation, and columns for each variable in the model.
We can access variable by providing ``variable path string``.
Here's an example of how to use it:

.. code::

    system = Success()
    model = Model(system)

    sim = Simulation(model, t_start=0, t_stop=100, num=200)
    sim.solve()
    dataframe = model.historian_df
    print(dataframe['root_system.system.item.namespace.variable']
