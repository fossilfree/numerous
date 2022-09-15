import numpy as np
import os
import pytest
from collections.abc import Callable
from scipy.integrate import solve_ivp, OdeSolution
from numba.core.registry import CPUDispatcher
from numerous.engine.simulation.solvers.numerous_solver.interface import SolverInterface, ModelInterface, SolveEvent, \
    jithelper
from numerous.engine.simulation.solvers.numerous_solver.numerous_solver import Numerous_solver

ABSTOL = 1e-6
RELTOL = 1e-6

class Solution:
    def __init__(self):
        self.results = []

    def add(self, t, y):
        self.results.append(np.append(t,y))

@jithelper
class SimpleModelInterface(ModelInterface):
    def __init__(self, n_tanks=10, start_volume=10, k=1):
        self.y = np.zeros(n_tanks)
        self.y[0] = start_volume
        self.k = k

    def get_deriv(self, t: float) -> np.array:
        y_dot = np.zeros(len(self.y))
        for tank in range(len(self.y)):
            if tank == 0:
                inlet = 0
            else:
                inlet = self.k*self.y[tank-1]

            if tank == len(self.y)-1:
                outlet = 0
            else:
                outlet = self.k*self.y[tank]

            diff = inlet-outlet
            y_dot[tank] = diff
        return y_dot

    def set_states(self, states: np.array) -> None:
        self.y = states

    def get_states(self):
        return self.y

    def historian_update(self, t: float) -> SolveEvent:
        return SolveEvent.Historian

    def fun(self, t, y):
        self.set_states(y)
        return self.get_deriv(t)


class ExampleSolverInterface(SolverInterface):
    def __init__(self, interface: SimpleModelInterface):
        self.model = interface
        self.sol = Solution()

    def handle_solve_event(self, event_id: SolveEvent, t: float):
        if event_id == SolveEvent.Historian:
            y = self.model.get_states()
            self.sol.add(t, y)

@pytest.fixture
def get_interface() -> Callable[[bool], ExampleSolverInterface]:
    def fn(jit=False):
        os.environ["_NO_JIT"] = "0" if jit else "1"

        modelinterface: [SimpleModelInterface, CPUDispatcher] = SimpleModelInterface()
        interface = ExampleSolverInterface(interface=modelinterface)
        return interface
    yield fn

@pytest.fixture
def get_timerange() -> (np.array, float):
    t_start = 0
    t_end = 100
    dt = 1
    timerange = np.append(np.arange(t_start, t_end, dt), t_end)
    yield timerange, dt

@pytest.fixture
def solve_ivp_results() -> Callable[[np.array, callable, str, np.array], OdeSolution]:
    def fn(y0, fun, method, timerange):
        if method == 'LM':
            method = 'BDF'

        sol = solve_ivp(fun, (timerange[0], timerange[-1]), y0, method=method, t_eval=timerange,
                        rtol=RELTOL, atol=ABSTOL)
        return sol
    yield fn


@pytest.mark.parametrize("method", ["BDF"])
@pytest.mark.parametrize("jit", [False])
def test_numerous_solver(get_interface: get_interface, solve_ivp_results: solve_ivp_results,
                         get_timerange: get_timerange,
                         method, jit):

    interface = get_interface(jit)
    y0 = interface.model.get_states()

    timerange, dt = get_timerange

    num_solver = Numerous_solver(timerange, dt, interface=interface,
                             y0=y0, numba_compiled_solver=jit, method=method, atol=ABSTOL, rtol=RELTOL)

    num_solver.solve()
    num_results = np.array(interface.sol.results).T[1:]

    scipy_results = solve_ivp_results(y0, interface.model.fun, method, timerange)

    for tank, (scpy, num) in enumerate(zip(scipy_results.y, num_results)):
        assert pytest.approx(scpy, abs=ABSTOL*10, rel=RELTOL*10) == num, f"results differ for tank {tank}"


