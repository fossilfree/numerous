import time
import math
from collections import namedtuple

import numpy as np
from numba import njit
from copy import deepcopy

from numerous.engine.simulation.solvers.base_solver import BaseSolver
from numerous.utils import logger as log
from .solver_methods import BaseMethod, RK45, Euler, BDF
from .interface import SolverInterface
from .solve_states import SolveEvent, SolveStatus

solver_methods = {'RK45': RK45, 'Euler': Euler, 'BDF': BDF}

Info = namedtuple('Info', ['status', 'event_id', 'step_info', 'initial_step', 'dt', 't', 'y', 'order_', 'roller',
                           'solve_state', 'ix_eval', 'g', 'step_converged', 'event_trigger'])

try:
    FEPS = np.finfo(1.0).eps
except AttributeError:
    FEPS = 2.220446049250313e-16


class Numerous_solver(BaseSolver):

    def __init__(self, time_, delta_t, interface: SolverInterface, y0, numba_compiled_solver,
                 **kwargs):
        super().__init__()

        self.time = time_
        self.delta_t = delta_t
        self.interface = interface
        self.g = self.interface.model.get_event_results(time_[0], y0)

        self.diff_function = interface.model.get_deriv
        self.numba_compiled_solver = numba_compiled_solver
        self.number_of_events = len(self.g) if all(self.g) else 0
        self.interface.model.pre_step(time_[0])
        self.interface.model.set_states(y0)
        self.f0 = self.diff_function(time_[0])
        solve_event_id = self.interface.model.historian_update(time_[0])
        self.interface.handle_solve_event(event_id=solve_event_id, t=time_[0])

        self.options = kwargs
        self.info = None


        odesolver_options = {'longer': kwargs.get('longer', 1.2), 'shorter': kwargs.get('shorter', 0.8),
                             'min_step': kwargs.get('min_step', 10 * FEPS), 'strict_eval': True,
                             'max_step': kwargs.get('max_step', np.inf), 'first_step': kwargs.get('first_step', None),
                             'atol': kwargs.get('atol', 1e-6), 'rtol': kwargs.get('rtol', 1e-3),
                             'submethod': kwargs.get('submethod', None),
                             }

        self.method_options = odesolver_options
        self.y0 = y0
        try:
            try:
                self.method = solver_methods[kwargs.get('method', 'RK45')]
            except KeyError:
                raise ValueError(f"Unknown method {kwargs.get('method', 'RK45')}, "
                                 f"allowed methods: {list(solver_methods.keys())}")
            self._method = self.method(self, **self.method_options)
            assert issubclass(self.method, BaseMethod), f"{self.method} is not a BaseMethod"
        except Exception as e:
            raise e

        # Generate the solver
        if numba_compiled_solver:
            self._non_compiled_solve = njit(self.generate_solver())
            self._solve = self.compile_solver()

        else:
            self._solve = self.generate_solver()

        self.interface.model.run_event_action(0, 0)

    def generate_solver(self):
        def _solve(interface, _solve_state, initial_step, dt_0, order, order_, roller, strict_eval,
                   min_step, max_step, step_integrate_, g, number_of_events, step_converged, event_trigger,
                   t0=0.0, t_end=1000.0, t_eval=np.linspace(0.0, 1000.0, 100), ix_eval=1, event_tolerance=1e-6):

            # Init t to t0
            imax = int(100)
            step_info = 0
            t = t0
            t_start = t0
            dt = dt_0
            interface.pre_step(t)
            y = interface.get_states()

            solve_status = SolveStatus.Running
            solve_event_id = SolveEvent.NoneEvent

            t_previous = t0
            y_previous = np.copy(y)

            # Define event derivatives, values and event guess times
            te_array = np.zeros(3)

            def is_internal_historian_update_needed(t_next_eval, t):
                if abs(t_next_eval - t) < 100 * FEPS:
                    return True
                return False

            def handle_converged(t, dt, ix_eval, t_next_eval):

                solve_event_id = SolveEvent.NoneEvent

                if is_internal_historian_update_needed(t_next_eval, t):
                    solve_event_id = interface.historian_update(t)
                    if strict_eval:
                        te_array[1] = t_next_eval = t_eval[ix_eval + 1] if ix_eval + 1 < len(t_eval) else t_eval[-1]
                    else:
                        t_next_eval = t_eval[ix_eval + 1] if ix_eval + 1 < len(t_eval) else t_eval[-1]
                    ix_eval += 1
                    te_array[0] = t + dt

                t_start = t
                t_new_test = np.min(te_array)

                return solve_event_id, ix_eval, t_start, t_next_eval, t_new_test

            def add_ring_buffer(t_, y_, rb, o):

                if o == order:
                    y_temp = rb[2][:, :]
                    t_temp = rb[1]
                    rb[1][0:order - 1] = t_temp[1:order]
                    rb[2][0:order - 1, :] = y_temp[1:order, :]

                o = min(o + 1, order)
                rb[1][o - 1] = t_
                rb[2][o - 1, :] = y_

                return o

            def get_order_y(rb, order):
                y = rb[2][0:order, :]
                return y

            # 0 index is used to keep next time step defined by solver
            te_array[0] = t
            # 1 index is used to keep next time to eval/save the solution

            te_array[1] = t_eval[ix_eval] + dt if strict_eval else np.inf
            t_next_eval = t_eval[ix_eval]

            event_ix = -1
            t_event = t
            y_event = y
            t_event_previous = -1
            time_event_ix, t_next_time_event = interface.get_next_time_event(t)
            if time_event_ix > 0:
                te_array[2] = t_next_time_event
            else:
                te_array[2] = np.max(te_array)

            while solve_status != SolveStatus.Finished:
                # updated events time estimates
                # # time acceleration
                if not step_converged:
                    if min_step > dt:
                        raise ValueError('dt shortened below min_dt')
                    te_array[0] = t_previous + dt
                elif step_converged and not event_trigger:
                    dt = min(max_step, dt)
                    te_array[0] = t + dt
                else:  # event
                    te_array[0] = t_event

                # Determine new test - it should be the smallest value requested by events, eval, step
                t_new_test = np.min(te_array)

                # Determine if rollback is needed
                # check if t_new_test is forward
                # no need to roll back if time-step is decreased, as it's not a failed step
                if (t_new_test < t) or (not step_converged):
                    # Need to roll back!
                    t_start = t_previous
                    y = y_previous

                    if t_new_test < t_start:
                        # t_new_test = t_rollback
                        # TODO: make more specific error raising here!
                        raise ValueError('Cannot go back longer than rollback point!')

                dt_ = min([t_next_eval - t_start, t_new_test - t_start])

                if order_ == 0:
                    order_ = add_ring_buffer(t, y, roller, order_)

                # solve from start to new test by calling the step function
                t, y, step_converged, step_info, _solve_state, factor = step_integrate_(interface,
                                                                                        t_start,
                                                                                        dt_, y,
                                                                                        get_order_y(roller, order_),
                                                                                        order_,
                                                                                        _solve_state)

                dt *= factor
                event_trigger = False
                t_events = np.zeros(number_of_events) + t
                y_events = np.zeros((len(y), number_of_events))

                def sol(t, t_r, y_r):
                    yi = np.zeros(len(y))
                    tv = np.append(roller[1][0:order], t_r)
                    yv = np.append(roller[2][0:order], y_r)
                    yv = yv.reshape(order + 1, len(y)).T
                    for i, yvi in enumerate(yv):
                        yi[i] = np.interp(t, tv, yvi)
                    return yi

                def check_event(event_fun, ix, t_previous, y_previous, t, y):
                    t_l = t_previous
                    y_l = y_previous
                    e_l = event_fun(t_l, y_l)[ix]
                    t_r = t
                    y_r = y
                    e_r = event_fun(t_r, y_r)[ix]
                    status = 0
                    if np.sign(e_l) == np.sign(e_r):
                        return status, t, y
                    i = 0
                    t_m = (t_l + t_r) / 2
                    y_m = sol(t_m, t, y)

                    while status == 0:  # bisection method
                        e_m = event_fun(t_m, y_m)[ix]
                        if np.sign(e_l) != np.sign(e_m):
                            t_r = t_m
                        elif np.sign(e_r) != np.sign(e_m):
                            t_l = t_m
                        if abs(e_m) < 1e-6 or abs(t_l - t_r) < 1e-6:
                            status = 1
                        if i > imax:
                            status = -1
                        t_m = (t_l + t_r) / 2
                        y_m = sol(t_m, t, y)

                    return status, t_r, sol(t_r, t, y)

                if step_converged:
                    time_event_ix, t_next_time_event = interface.get_next_time_event(t)
                    if time_event_ix > 0:
                        te_array[2] = t_next_time_event
                    else:
                        te_array[2] = np.max(te_array)

                    if number_of_events > 0:

                        g_new = interface.get_event_results(t, y)

                        up = (g <= 0) & (g_new >= 0) & (interface.get_event_directions() == 1)
                        down = (g >= 0) & (g_new <= 0) & (interface.get_event_directions() == -1)
                        g = g_new

                        for ix in np.concatenate((np.argwhere(up), np.argwhere(down))):
                            eps = 1.e-6  # for case to t_event = t
                            status, t_event, y_event = check_event(interface.get_event_results, ix[0],
                                                                   t_previous, y_previous, t, y)
                            t_events[ix[0]] = t_event - eps
                            y_events[:, ix[0]] = y_event

                if number_of_events > 0 and min(t_events) < t:
                    event_trigger = True
                    event_ix = np.argmin(t_events)
                    t_event = t_events[event_ix]
                    y_event = y_events[:, event_ix]
                    g = interface.get_event_results(t_event, y_event)

                if not event_trigger and step_converged:
                    y_previous = y
                    t_previous = t
                    if abs(t - t_next_time_event) < 1e-6:
                        interface.set_states(y)
                        interface.run_time_event_action(t, time_event_ix)  # todo: use interface to get y.
                        y_previous = interface.get_states()

                if event_trigger:
                    # Improve detection of event

                    if abs(t_event - t_event_previous) > event_tolerance:
                        t_event_previous = t_event
                        step_converged = False  # roll back and refine search
                        dt = initial_step
                        g = interface.get_event_results(t, y_previous)
                    else:
                        interface.set_states(y_event)
                        interface.run_event_action(t_event, event_ix)

                        y_previous = interface.get_states()
                        t_previous = t_event

                        # TODO: Update solve in solver_methods with new solve state after changing states due to events

                        solve_event_id = interface.post_event(t_event)
                        # Immediate rollback in case of exit
                        t = t_previous
                        y = y_previous

                        if solve_event_id != SolveEvent.NoneEvent:
                            break

                if step_converged:
                    interface.set_states(y)
                    solve_event_id = interface.post_step(t)
                    if solve_event_id != SolveEvent.NoneEvent:
                        break

                    order_ = add_ring_buffer(t, y, roller, order_)

                    solve_event_id, ix_eval, t_start, t_next_eval, t_new_test = \
                        handle_converged(t, dt, ix_eval, t_next_eval)

                    if abs(t - t_end) < 100 * FEPS:
                        solve_status = SolveStatus.Finished
                        break

                    if solve_event_id != SolveEvent.NoneEvent:
                        break

            return Info(status=solve_status, event_id=solve_event_id, step_info=step_info,
                        dt=dt, t=t, y=np.ascontiguousarray(y), order_=order_, roller=roller, solve_state=_solve_state,
                        ix_eval=ix_eval, g=g, initial_step=initial_step, step_converged=step_converged,
                        event_trigger=event_trigger)

        return _solve

    def compile_solver(self):

        log.info("Compiling Numerous Solver")
        generation_start = time.time()

        argtypes = []

        max_step = self.method_options.get('max_step')
        min_step = self.method_options.get('min_step')

        strict_eval = self.method_options.get('strict_eval', True)

        order = self._method.order
        initial_step = min_step

        step_integrate_ = self._method.step_func
        roller = self._init_roller(order)
        order_ = 0

        args = (self.interface.model,
                self._method.get_solver_state(len(self.y0)),
                initial_step,
                initial_step,
                order,
                order_,
                roller,
                strict_eval,
                min_step,
                max_step,
                step_integrate_,
                self.g,
                self.number_of_events,
                False,
                False,
                self.time[0],
                self.time[-1],
                self.time,
                1,
                self.method_options.get('atol')
                )
        for a in args:
            argtypes.append(self._non_compiled_solve.typeof_pyval(a))
        # Return the solver function

        _solve = self._non_compiled_solve.compile(tuple(argtypes))

        generation_finish = time.time()
        log.info(f"Solver compiled, compilation time: {generation_finish - generation_start}")

        return _solve

    def _init_roller(self, order):
        n = order + 2
        rb0 = np.zeros((n, len(self.y0)))
        roller = (n, np.zeros(n), rb0)
        return roller

    def use_no_state_solver(self):
        states = self.interface.model.get_states()
        if states.shape[0] == 0:
            return True

    def _no_state_solver_step(self, t, dt):
        '''
        This method calculates the model variables using mappings, which are pushed through using the .func method of
        the numba_model class. It assumes that the first step has already been saved in the historian.
        '''
        solve_event_id = self.interface.model.post_step(t)
        self.interface.handle_solve_event(solve_event_id, t)

        states = self.interface.model.get_states()
        t += dt
        if t > self.time[-1]:
            return

        self.interface.model.post_step(t)
        self.interface.model.set_states(states)
        self.interface.model.get_deriv(t)  # update mappings
        self.interface.model.historian_update(t)

    def _init_solve(self, info=None):

        max_step = self.method_options.get('max_step')
        min_step = self.method_options.get('min_step')
        rtol = self.method_options.get('rtol')
        atol = self.method_options.get('atol')
        strict_eval = self.method_options.get('strict_eval')
        step_integrate_ = self._method.step_func
        order = self._method.order

        if not info:
            t_start = self.time[0]

            # Call the solver

            y0 = deepcopy(self.y0)

            # Set options

            initial_step = self._method.initial_step

            dt = initial_step

            # figure out solve_state init
            solve_state = self._method.get_solver_state(len(y0))

            roller = self._init_roller(order)
            order_ = 0
            g = self.g
            step_converged = False
            event_trigger = False
        else:

            dt = self.info.dt  # internal solver step size
            order_ = self.info.order_
            roller = self.info.roller
            solve_state = self.info.solve_state
            g = self.info.g
            initial_step = self.info.initial_step
            step_converged = self.info.step_converged
            event_trigger = self.info.event_trigger

        return dt, strict_eval, step_integrate_, solve_state, roller, order_, order, initial_step, min_step, \
               max_step, atol, g, step_converged, event_trigger

    def solve(self):
        """
        solve the model.

        Returns
        -------
        Solution : 'OdeSolution'
                returns the most recent OdeSolution from scipy

        """
        self.result_status = "Success"
        self.sol = None
        log.info('Solve started')

        if self.use_no_state_solver():
            log.info("No states in model. Using no-state solver.")
            dt = self.time[1] - self.time[0]
            for t in self.time[0:-1]:
                self._no_state_solver_step(t, dt)
            return self.sol, self.result_status

        self._solver(self.time)

        log.info("Solve finished")
        return self.sol, self.result_status

    def solver_step(self, t, delta_t=None):

        if self.use_no_state_solver():
            self._no_state_solver_step(t, delta_t)
            return t + delta_t, t + delta_t

        if delta_t is None:
            delta_t = self.delta_t

        n = math.floor(t / delta_t + FEPS * 100) + 1
        t_eval = np.linspace(t, n * delta_t, 2)
        t_end = t_eval[-1]

        info = self._solver(t_eval, info=self.info)
        self.info = info

        return t_end, self.info.t

    def _solver(self, t_eval, info=None):

        dt, strict_eval, step_integrate_, solve_state, roller, order_, order, initial_step, min_step, max_step, \
            atol, g, step_converged, event_trigger = self._init_solve(info)

        t_start = t_eval[0]
        t_end = t_eval[-1]

        info = self._solve(self.interface.model,
                           solve_state, initial_step, dt, order, order_, roller, strict_eval, min_step,
                           max_step, step_integrate_, g,
                           self.number_of_events,
                           step_converged,
                           event_trigger,
                           t_start, t_end,
                           t_eval, 1, atol)

        while info.status == SolveStatus.Running:
            self.interface.handle_solve_event(info.event_id, info.t)

            info = self._solve(self.interface.model,
                               info.solve_state, initial_step, info.dt, order, info.order_, info.roller, strict_eval,
                               min_step, max_step, step_integrate_, info.g,
                               self.number_of_events,
                               info.step_converged,
                               info.event_trigger,
                               info.t, t_end,
                               t_eval, info.ix_eval, atol)

        self.interface.handle_solve_event(info.event_id, info.t)
        return info

    def register_endstep(self, __end_step):
        self.__end_step = __end_step
