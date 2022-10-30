
Ideas:
- General idea - specify more info on the class for simpler notation, less boilerplate, enable static evaluation
- Define variables in a class extending a Scope class
-- Static check of variables for an time
-- Type hinting and autocompletion in equations
- Scope is instanciated on class and assigned to class variable
-- namespace for each scope can be automatically created and registered
-- equation decorator takes the scope as argument and can add equation to namespace automatically
-- scope class works as typehint for scope argument for equation - allows autocompletion in equation
- static mappings can be made on class
- sub items are specified in a class extending ItemSpec
-- autocompletion of items in mappings and __init__
-- items automatically registered on the active subsystem they are made in
-- automated check if items have been assigned - raise error otherwise