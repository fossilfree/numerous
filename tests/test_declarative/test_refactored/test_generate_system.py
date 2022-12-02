

import pytest
import logging

logging.basicConfig(level=logging.DEBUG)

from numerous.declarative.generate_system import generate_system
from numerous.declarative import Module, ItemsSpec, ScopeSpec, Connector, Parameter, set_value_from, get_value_for, EquationSpec

@pytest.fixture
def TestModule():
    class TestModule(Module):

        class Items(ItemsSpec):
            a: Module
            b: Module

        items = Items()

        class Variables(ScopeSpec):

            var1 = Parameter(0)
            var2 = Parameter(0)
            var3 = Parameter(0)
            var4 = Parameter(10)

        variables = Variables()

        variables.var3 += variables.var4

        connector = Connector(
            var1=set_value_from(variables.var1),
            mod1=set_value_from(items.a)
        )

        connector2 = Connector(
            var1=get_value_for(variables.var2),
            mod1=get_value_for(items.b)

        )

        connector >> connector2
        
        def __init__(self):
            super(TestModule, self).__init__()
            self.items.a = Module()

        @EquationSpec(variables)
        def eq(self, scope:Variables):
            scope.var1 = 1.0


    return TestModule

def test_generate_module(TestModule):

    test_module = TestModule()

    system = generate_system("sys", test_module)


    from numerous.engine import simulation
    from numerous.engine import model
    from matplotlib import pyplot as plt

    if True:
        # Define simulation
        s = simulation.Simulation(model.Model(system, use_llvm=False), t_start=0, t_stop=500.0, num=1000, num_inner=100,
                                  max_step=1)
        # Solve and plot

        s.solve()

        s.model.historian_df[["sys.variables.var1", "sys.variables.var2"
                             ]
        ].plot()
        s.model.historian_df[["sys.variables.var3", "sys.variables.var4",
                              ]
        ].plot()

        # print()
        plt.show()
        plt.interactive(False)