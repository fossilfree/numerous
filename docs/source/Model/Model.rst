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



Model cloning and caching
^^^^^^^^^^^^^^^^^^

Cloning a model allows you to create a copy of an existing model, which can then be used for further simulations
or analysis without modifying the original model. This can be useful when you want to run multiple simulations
with slightly different parameter values or initial conditions. To clone a model, you can call the clone()
method on an existing model object, and then pass the resulting object to a new simulation.
Caching a model allows you to save the results of a simulation to disk, so that they can be reused later
without having to re-run the simulation. This can be useful when you have a large or computationally
expensive model and you want to avoid running the simulation multiple times. To cache a model, you can
call the cache() method on a simulation object, and then pass the resulting object to a new simulation.
The clone() and cache() methods are both optional arguments of Model class and can be passed as True or
False during instantiation of the Model. By default, both clone and cache arguments are set to False, and
the model is not clonable or cacheable.
When caching is enabled, the states of the model will be saved to the cache file before the simulation starts.
And when the simulation is run again, the model will load the states from the cache file and continue the
simulation from there.
When using cloning and caching together, it is important to note that cloning a cached model will also
clone the cache file. This means that if you make changes to the cloned model, the original model
and its cache file will not be affected.
It is important to note that caching and cloning of the model will increase the memory usage of
the engine and thus it should be used with care, especially when dealing with large models.

.. code::

    #creating a model
    system = S2N("S2", 2)
    model = Model(system)

    #cloning the model
    cloned_model = model.clone()


    #cloning the cached model
    cloned_cached_model = simulation.model.clone()

As a best practice, it is recommended to use caching and cloning as needed, and avoid using them when not necessary.
This can help to optimize the performance and memory usage of the engine.


Getting results of the computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
historian_df



Creating aliases for variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Having many nested subsytems can make path to the variable long or non-unique.
If we want to track some specific variable we can add alies to it.

