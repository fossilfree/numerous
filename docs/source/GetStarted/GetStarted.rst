Get Started
===================

To install using pip use `pip install numerous-engine`


You can get started quickly here with a simple example:

```python
from numerous.engine import model, simulation
from numerous.examples.dampened_oscillator.dampened_oscillator import OscillatorSystem
#Define simulation
s = simulation.Simulation(
     model.Model(OscillatorSystem('system')),
    t_start=0, t_stop=10, num=100, num_inner=100, max_step=0.1
)
#Solve
s.solve()
simulation_result = s.model.historian_df
```


Or follow one of our comprehensive tutorials:

 * [Bouncing_Ball](https://github.com/fossilfree/numerous/blob/master/examples/Bouncing_Ball/Bouncing%20Ball%20Example.ipynb)
	a model of a ball dropping from a certain height and bouncing off the ground, until it finally comes to rest
 * [Two_Tanks_System](https://github.com/fossilfree/numerous/blob/master/examples/Two_Tanks_System/TwoTanks_System.ipynb)
	two tanks are placed on top of each other and connected by a valve.