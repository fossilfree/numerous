![Numerous Logo](./docs/source/_static/img/logo_cropped_numerous_transparent-back.svg)
---------------------------------------------------------------------------------------------------

Numerous — an object-oriented modelling and simulation engine 
================================================================

Reasons for developing a python-based object-oriented simulation engine 

The arguments for a python simulation engine are numerous: 

* In order to take advantage of cloud based computing power a run-anywhere solution is preferred
* Open-source to eliminate the need of complex licenses for deploying thousands of simulations
* Direct connection with machine-learning and artificial intelligence libraries through the most popular programming language for data science
* Leveraging packages from the vast community seamlessly


Philosophy and Motivation for Object-Oriented Modelling
--------------------------------------------------------

As systems becomes complex the number of equations and variables grow fast and the overview is lost for the model developer. The idea behind this engine is to allow the model developer focus on one familiar object at a time and setup simulations for validation – and then combine these objects together to form complex interacting systems in a simple way where all the general tedious work is handled by the engine. 

Quick start
--------------------------------------------------------
To install using pip use `pip install numerous-engine`


You can get started quickly here with a simple example:

```python
from numerous.engine import model, simulation
from numerous.examples.dampened_oscillator import OscillatorSystem

#Define simulation
s = simulation.Simulation(
     model.Model(OscillatorSystem('system')),
    t_start=0, t_stop=10, num=100, num_inner=100, max_step=0.1
)
#Solve and plot
s.solve()
s.model.historian.df.plot()
```


Or follow one of our comprehensive tutorials: 

 * [Bouncing_Ball](https://github.com/fossilfree/numerous/blob/master/examples/Bouncing_Ball/Bouncing%20Ball%20Example.ipynb)
	a model of a ball dropping from a certain height and bouncing off the ground, until it finally comes to rest
 * [Two_Tanks_System](https://github.com/fossilfree/numerous/blob/master/examples/Two_Tanks_System/TwoTanks_System.ipynb)
	two tanks are placed on top of each other and connected by a valve.

Documentation can be found on [readthedocs](https://numerous.readthedocs.io/).

Or you can get familiar with the concepts we have used to abstract away building complex interacting systems right inside python. 
