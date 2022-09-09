import numpy as np
import numba as nb
from numba.core.registry import CPUDispatcher
from numba.experimental import jitclass
from numerous.engine.simulation.solvers.numerous_solver.interface import SolverInterface, ModelInterface, SolveEvent
from numerous.engine.simulation.solvers.numerous_solver.numerous_solver import Numerous_solver

def _jitclass(jit):
    def wrapper(spec):
        if jit:
            return jitclass(spec)
        else:
            def passthrough(fun):
                return fun
            return passthrough
    return wrapper

class ExampleModel:
    def __init__(self, n_tanks=10, start_volume=10, k=1):
        self.n_tanks = n_tanks
        self.y = np.zeros(n_tanks)
        self.y[0] = start_volume
        self.y_dot = np.zeros(n_tanks)
        self.k = k

    def flow_in(self, tank: int):
        flow_in = 0  # No input at tank 1
        if tank > 0:
            flow_in = self.k * self.y[tank-1]

        return flow_in

    def flow_out(self, tank: int):
        if tank == self.n_tanks-1:  # No output at tank n_tanks
            return 0
        return self.k * self.y[tank]

    def diff(self, tank: int):
        y_dot = self.flow_in(tank)-self.flow_out(tank)
        return y_dot

class ExampleModelInterface(ModelInterface):
    def __init__(self, model: [CPUDispatcher, ExampleModel]):
        self.model = model

    def get_deriv(self, t: float) -> np.array:
        y_dot = np.zeros(self.model.n_tanks)
        for tank in range(self.model.n_tanks):
            y_dot[tank] = self.model.diff(tank)
        return y_dot

    def get_states(self) -> np.array:
        return self.model.y

    def set_states(self, states: np.array) -> None:
        self.model.y = states

    def historian_update(self, t: float) -> SolveEvent:
        return SolveEvent.Historian

    def get_next_time_event(self, t) -> tuple[int, float]:
        return -1, -1

class Solution:
    def __init__(self):
        self.results = []

    def add(self, t, y):
        self.results.append(np.append(t,y))


class ExampleSolverInterface(SolverInterface):
    def __init__(self, modelinterface: ExampleModelInterface):
        super(ExampleSolverInterface, self).__init__(modelinterface=modelinterface)
        self.sol = Solution()

    def handle_solve_event(self, event_id: SolveEvent, t: float):
        if event_id == SolveEvent.Historian:
            y = self.model.get_states()
            self.sol.add(t, y)

def generate_interface(jit=True):
    jitclass_ = _jitclass(jit=jit)

    modelspec = [("n_tanks", nb.int64), ("start_volume", nb.float64), ("k", nb.float64),
                 ("y", nb.float64[:]),
                 ("y_dot", nb.float64[:])]

    @jitclass_(spec=modelspec)
    class ModelWrapper(ExampleModel):
        pass

    model: [CPUDispatcher, ExampleModel] = ModelWrapper()
    interfacespec = [("model", model._numba_type_ if hasattr(model, '_numba_type_') else model)]

    @jitclass_(spec=interfacespec)
    class ModelInterfaceWrapper(ExampleModelInterface):
        pass

    modelinterface = ModelInterfaceWrapper(model=model)
    interface = ExampleSolverInterface(modelinterface=modelinterface)
    return interface

def numerous_solver_test(jit=True):
    interface = generate_interface(jit)
    y0 = interface.model.get_states()

    t_start = 0
    t_end = 100
    dt = 1
    timerange = np.append(np.arange(t_start, t_end, dt), t_end)

    solver = Numerous_solver(timerange, dt, interface=interface,
                             y0=y0, numba_compiled_solver=jit)
    solver.solve()

    results = interface.sol.results
    a=1


if __name__ == "__main__":
    numerous_solver_test(jit=True)