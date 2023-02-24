from numerous.multiphysics.equation_base import EquationBase
from numerous.engine.system.item import Item
from numerous.multiphysics.equation_decorators import Equation
from numerous.utils.data_loader import InMemoryDataLoader
from numerous.engine.system.subsystem import Subsystem
from numerous.engine import model, simulation
from numerous.engine.simulation.solvers.base_solver import SolverType
from numerous.engine.system.external_mappings import ExternalMappingElement, InterpolationType

import pandas as pd


class Time(Item, EquationBase):
    def __init__(self,
                 tag='time'):

        super(Time, self).__init__(tag)

        dn = self.create_namespace('default')

        self.add_state('t', init_val=0)
        self.add_parameters({'p1': 0})

        dn.add_equations([self])

    @Equation()
    def diff(self, scope):
        scope.t_dot = 1


class TestSystem(Subsystem):

    def __init__(self, tag='sys', data_loader=None, external_mappings=None):
        super().__init__(tag, external_mappings=external_mappings, data_loader=data_loader)

        time_ = Time()

        self.register_items([time_], tag="simples")


if __name__ == '__main__':
    n_hours = 10
    len_data = n_hours + 1
    df = pd.DataFrame({'p1': range(len_data), 't': range(len_data)})

    data_loader = InMemoryDataLoader(df)

    external_mappings = [
        ExternalMappingElement(0, 't', 0, 3600, {'sys.time.default.p1': ('p1', InterpolationType.LINEAR)}),
    ]

    sys = TestSystem(external_mappings=external_mappings, data_loader=data_loader)
    m = model.Model(sys, use_llvm=True)

    # Define simulation
    s = simulation.Simulation(
        m,
        t_start=0, t_stop=n_hours * 3600, num=n_hours, num_inner=1, max_step=3600,
        solver_type=SolverType.NUMEROUS, method="RK45"
    )

    s.solve()

    assert (s.model.historian_df['sys.time.default.p1'] == df['p1']).all(), "All elements should be equal value!"
