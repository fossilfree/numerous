Simulation
==================

The Simulation class in the :class:`numerous.engine.simulation` module is used to run the simulation of a system over time.
To create a simulation, a Model object must first be instantiated, which represents the system being  completed and compiled.
Once a Model object is created, it can be passed as the first argument to the Simulation class, along with the start and
stop time of the simulation, the number of time steps to take, and the number of inner steps for the solver to use.
Here is an example of how to create a simulation for a Subsystem object called system with a start time of 0, a stop
time of 2, and a total of 2 time steps:

.. code::


    from numerous.engine.model import Model
    from numerous.engine.simulation import Simulation

    model = Model(system)
    simulation = Simulation(model, t_start=0, t_stop=2, num=2)
    simulation.solve()


You can also pass additional parameters to the ``Simulation`` constructor that represent options for numerous solver.
Once the simulation is created, you can use the ``solve()`` method to run the simulation and solve the system of equations.
The result of the simulation can be accessed through the model attribute of the ``Simulation`` object,
in ``historian_df`` field of the model.
You can also use the ``reset(self, t_start:float)`` method to resets the simulation and model states to their initial values.
The Numerous Engine allows users to create a step solver simulation by
calling the ``step_solve(t_start:float, step_size:float)``  method.
To make a single simulation step.

.. code::


    from numerous.engine.model import Model
    from numerous.engine.simulation import Simulation

    model = Model(system)
    simulation = Simulation(model, t_start=0, t_stop=2, num=2)
    simulation.step_solve(t_start=0,  step_size=0.1)

