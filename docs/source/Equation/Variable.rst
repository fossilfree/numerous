Variable
====================
These are the variables the equation function will operate on.

There are several types of variables:

-States
-State derivatives
-Parameters
-Constants

States are the dependent variables of the system. The solver will integrate the system using the state derivates and store the result in the states.

Parameters can be used to store intermediate calculations or parameters for the system that can be changed during the simulation.

Constants are initialized with a fixed initial value and will not change over the course of the simulation.

Scope
====================
Each equation instance will have a scope consisting of the variables defined in the equation class with the current values of each variable in the simulation. The equation functions can only operate on the variables in its scope.