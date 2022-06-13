import logging
import time
from collections import namedtuple
from enum import IntEnum, unique

import numpy as np
from numba import njit

from numerous.engine.simulation.solvers.base_solver import BaseSolver
from .solver_methods import BaseMethod, RK45, Euler

solver_methods = {'RK45': RK45, 'Euler': Euler}

Info = namedtuple('Info', ['status', 'event_id', 'step_info', 'dt', 't', 'y', 'order_', 'roller', 'solve_state',
                           'ix_eval'])


@unique
class SolveStatus(IntEnum):
    Running = 0
    Finished = 1


@unique
class SolveEvent(IntEnum):
    NoneEvent = 0
    Historian = 1
    ExternalDataUpdate = 2
    HistorianAndExternalUpdate = 3


class Numerous_solver(BaseSolver):

    def __init__(self, time_, delta_t, model, numba_model, num_inner, max_event_steps, y0, numba_compiled_solver,
                 events,
                 event_directions,
                 timestamp_events,
                 **kwargs):
        super().__init__()

        def get_variables_modified(y_):
            old_states = numba_model.get_states()
            numba_model.set_states(y_)

            vars = numba_model.read_variables().copy()
            numba_model.set_states(old_states)
            return vars

        self.events = events[0][0]
        self.event_directions = event_directions
        self.actions = events[1][0]
        self.timestamps = timestamp_events[1]
        self.timestamps_actions = timestamp_events[0][0]
        # events value
        self.g = self.events(time_[0], get_variables_modified(y0))

        self.time = time_
        self.model = model
        self.num_inner = num_inner
        self.delta_t = delta_t
        self.numba_model = numba_model
        self.diff_function = numba_model.func
        self.numba_compiled_solver = numba_compiled_solver
        self.number_of_events = len(self.g)

        self.f0 = numba_model.func(time_[0], y0)
        self.max_event_steps = max_event_steps
        self.options = kwargs
        self.info = None

        feps = np.finfo(1.0).eps  # 2.220446049250313e-16#
        odesolver_options = {'longer': kwargs.get('longer', 1.2), 'shorter': kwargs.get('shorter', 0.8),
                             'min_step': kwargs.get('min_step', 10 * feps), 'strict_eval': True,
                             'max_step': kwargs.get('max_step', np.inf), 'first_step': kwargs.get('first_step', None),
                             'atol': kwargs.get('atol', 1e-6), 'rtol': kwargs.get('rtol', 1e-3),
                             'outer_itermax': kwargs.get('outer_itermax', 20),
                             'submethod': kwargs.get('submethod', None),
                             }

        self.method_options = odesolver_options
        try:
            try:
                self.method = solver_methods[kwargs.get('method', 'RK45')]
            except KeyError:
                raise ValueError(f"Unknown method {kwargs.get('method', 'RK45')}, allowed methods: {list(solver_methods.keys())}")
            self._method = self.method(self, **self.method_options)
            assert issubclass(self.method, BaseMethod), f"{self.method} is not a BaseMethod"
        except Exception as e:
            raise e

        self.y0 = y0

        def run_event_action(actions, time_, numba_model, action_id):
            modified_variables = actions(time_, numba_model.read_variables(), action_id)
            modified_mask = (modified_variables != numba_model.read_variables())
            for idx in np.argwhere(modified_mask):
                numba_model.write_variables(modified_variables[idx[0]], idx[0])

        # Generate the solver
        if numba_compiled_solver:
            self.run_event_action = njit(run_event_action)
            self._non_compiled_solve = njit(self.generate_solver())
            self._solve = self.compile_solver()

        else:
            self.run_event_action = run_event_action
            self._solve = self.generate_solver()

        self.run_event_action(self.actions, 0, numba_model, 0)

    def generate_solver(self):
        def _solve(numba_model, _solve_state, initial_step, order, order_, roller, strict_eval, outer_itermax,
                   min_step, max_step, step_integrate_, events, actions, g, number_of_events, event_directions,
                   run_event_action, timestamps, timestamp_actions,
                   t0=0.0, t_end=1000.0, t_eval=np.linspace(0.0, 1000.0, 100), ix_eval=1, event_tolerance=1e-6):

            # Init t to t0
            imax = 100
            step_info = 0
            t = t0
            dt = initial_step
            y = numba_model.get_states()
            feps = np.finfo(1.0).eps  # 2.220446049250313e-16
            t_previous = t0
            y_previous = np.copy(y)

            def get_event_id(t):
                if numba_model.is_store_required() and not numba_model.is_external_data_update_needed(t):
                    return SolveEvent.Historian
                elif not numba_model.is_store_required() and numba_model.is_external_data_update_needed(t):
                    return SolveEvent.ExternalDataUpdate
                elif numba_model.is_store_required() and numba_model.is_external_data_update_needed(t):
                    return SolveEvent.HistorianAndExternalUpdate
                else:
                    return SolveEvent.NoneEvent

            def get_variables_modified(y_):
                old_states = numba_model.get_states()
                numba_model.set_states(y_)

                vars = numba_model.read_variables().copy()
                numba_model.set_states(old_states)
                return vars

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

            # Define event derivatives, values and event guess times
            # edt = np.ones(max(1, n_events)) * tol + 2.0
            # et = np.zeros(n_events) if n_events > 0 else np.ones(1)
            te_array = np.zeros(2)
            # 0 index is used to keep next time step defined by solver
            te_array[0] = t
            # 1 index is used to keep next time to eval/save the solution

            te_array[1] = t_eval[ix_eval] + dt if strict_eval else np.inf
            t_next_eval = t_eval[ix_eval]

            terminate = False
            step_converged = True
            event_trigger = False
            event_ix = -1
            t_event = t
            y_event = y
            t_event_previous = -1
            solve_status = SolveStatus.Finished
            while not terminate:
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
                else:
                    # Since we didnt roll back we can update t_start and rollback
                    # Check if we should update history at t eval
                    if abs(t_next_eval - t) <= 10 * feps:
                        if not step_converged:
                            print("step not converged, but historian updated")
                        numba_model.historian_update(t)
                        if strict_eval:
                            te_array[1] = t_next_eval = t_eval[ix_eval + 1] if ix_eval + 1 < len(t_eval) else t_eval[-1]
                        else:
                            t_next_eval = t_eval[ix_eval + 1] if ix_eval + 1 < len(t_eval) else t_eval[-1]
                        ix_eval += 1

                        dt = min(max_step, dt)
                        te_array[0] = t + dt

                    t_start = t
                    t_new_test = np.min(te_array)
                    if t >= t_end:
                        solve_status = SolveStatus.Finished
                        break
                    if get_event_id(t) > 0:
                        solve_status = SolveStatus.Running
                        break

                    order_ = add_ring_buffer(t, y, roller, order_)

                dt_ = min([t_next_eval - t_start, t_new_test - t_start])
                # solve from start to new test by calling the step function
                t, y, step_converged, step_info, _solve_state, factor = step_integrate_(numba_model,
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
                    e_l = event_fun(t_l, get_variables_modified(y_l))[ix]
                    t_r = t
                    y_r = y
                    e_r = event_fun(t_r, get_variables_modified(y_r))[ix]
                    status = 0
                    if np.sign(e_l) == np.sign(e_r):
                        return status, t, y
                    i = 0
                    t_m = (t_l + t_r) / 2
                    y_m = sol(t_m, t, y)

                    while status == 0:  # bisection method
                        e_m = event_fun(t_m, get_variables_modified(y_m))[ix]
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
                    g_new = events(t, get_variables_modified(y))

                    up = (g <= 0) & (g_new >= 0) & (event_directions == 1)
                    down = (g >= 0) & (g_new <= 0) & (event_directions == -1)
                    g = g_new

                    for ix in np.concatenate((np.argwhere(up), np.argwhere(down))):
                        eps = 1.e-6  # for case to t_event = t
                        status, t_event, y_event = check_event(events, ix[0],
                                                               t_previous, y_previous, t, y)
                        t_events[ix[0]] = t_event - eps
                        y_events[:, ix[0]] = y_event

                if min(t_events) < t:
                    event_trigger = True
                    event_ix = np.argmin(t_events)
                    t_event = t_events[event_ix]
                    y_event = y_events[:, event_ix]
                    g = events(t_event, get_variables_modified(y_event))

                if not event_trigger and step_converged:
                    y_previous = y
                    t_previous = t
                    for event_ix, timestamps_ in enumerate(timestamps):
                        for timestamp in timestamps_:
                            if abs(timestamp - t) < 1e-6:
                                numba_model.set_states(y)
                                modified_variables = timestamp_actions(t, numba_model.read_variables(), event_ix)
                                modified_mask = (modified_variables != numba_model.read_variables())
                                for idx in np.argwhere(modified_mask):
                                    numba_model.write_variables(modified_variables[idx[0]], idx[0])
                                y_previous = numba_model.get_states()

                if event_trigger:
                    # Improve detection of event

                    if abs(t_event - t_event_previous) > event_tolerance:
                        t_event_previous = t_event
                        step_converged = False  # roll back and refine search
                        dt = initial_step
                        g = events(t, get_variables_modified(y_previous))
                    else:

                        numba_model.set_states(y_event)

                        run_event_action(actions, t_event, numba_model, event_ix)

                        y_previous = numba_model.get_states()
                        t_previous = t_event

                        # TODO: Update solve in solver_methods with new solve state after changing states due to events

                        numba_model.historian_update(t_event)
                        # Immediate rollback in case of exit
                        t = t_previous
                        y = y_previous

                if step_converged:
                    numba_model.map_external_data(t)
                    solve_event_id = get_event_id(t)
                    if solve_event_id > 0:
                        return Info(status=SolveStatus.Running, event_id=solve_event_id, step_info=step_info,
                                    dt=dt, t=t, y=np.ascontiguousarray(y), order_=order_, roller=roller,
                                    solve_state=_solve_state, ix_eval=ix_eval)

            return Info(status=solve_status, event_id=get_event_id(t), step_info=step_info,
                        dt=dt, t=t, y=np.ascontiguousarray(y), order_=order_, roller=roller, solve_state=_solve_state,
                        ix_eval=ix_eval)

        return _solve

    def compile_solver(self):

        logging.info("Compiling Numerous Solver")
        generation_start = time.time()

        argtypes = []

        max_step = self.method_options.get('max_step')
        min_step = self.method_options.get('min_step')

        strict_eval = self.method_options.get('strict_eval', True)
        outer_itermax = self.method_options.get('outer_itermax', 20)

        order = self._method.order
        initial_step = min_step

        step_integrate_ = self._method.step_func
        roller = self._init_roller(order)
        order_ = 0

        args = (self.numba_model,
                self._method.get_solver_state(len(self.y0)), initial_step,
                order, order_, roller, strict_eval, outer_itermax, min_step,
                max_step, step_integrate_,
                self.events,
                self.actions,
                self.g,
                self.number_of_events,
                self.event_directions,
                self.run_event_action,
                self.timestamps,
                self.timestamps_actions,
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
        logging.info(f"Solver compiled, compilation time: {generation_finish - generation_start}")

        return _solve

    def select_initial_step(self, nm, t0, y0, direction, order, rtol, atol):
        """Taken from scipy select initial step part of the ode solver package. Slightly modified


        Empirically select a good initial step.

        The algorithm is described in [1]_.

        Parameters
        ----------
        nm : Numbamodel - contains the function handles to call the RHS

        t0 : float
            Initial value of the independent variable.
        y0 : ndarray, shape (n,)
            Initial value of the dependent variable.
        direction : float
            Integration direction.
        order : float
            Error estimator order. It means that the error controlled by the
            algorithm is proportional to ``step_size ** (order + 1)`.
        rtol : float
            Desired relative tolerance.
        atol : float
            Desired absolute tolerance.

        Returns
        -------
        h_abs : float
            Absolute value of the suggested initial step.

        References
        ----------
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
               Equations I: Nonstiff Problems", Sec. II.4.
        """

        f0 = self.f0

        if y0.size == 0:
            return np.inf

        scale = atol + np.abs(y0) * rtol
        d0 = np.linalg.norm(y0 / scale)
        d1 = np.linalg.norm(f0 / scale)
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = y0 + h0 * direction * f0
        f1 = nm.func(t0 + h0 * direction, y1)
        d2 = np.linalg.norm((f1 - f0) / scale) / h0

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

        # Restore states in numba model
        nm.func(t0, y0)

        return min(100 * h0, h1)

    def _init_roller(self, order):
        n = order + 2
        rb0 = np.zeros((n, len(self.y0)))
        roller = (n, np.zeros(n), rb0)
        return roller

    def load_external_data(self, t):
        is_external_data = self.model.external_mappings.load_new_external_data_batch(t)
        external_mappings_numpy = self.model.external_mappings.external_mappings_numpy
        external_mappings_time = self.model.external_mappings.external_mappings_time
        max_external_t = self.model.external_mappings.t_max
        self.numba_model.is_external_data = is_external_data
        self.numba_model.update_external_data(external_mappings_numpy, external_mappings_time, max_external_t)

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
        logging.info('Solve started')
        # try:
        t_start = self.time[0]
        t_end = self.time[-1]

        # Call the solver
        from copy import deepcopy
        y0 = deepcopy(self.y0)

        # Set options

        max_step = self.method_options.get('max_step')
        min_step = self.method_options.get('min_step')
        rtol = self.method_options.get('rtol')
        atol = self.method_options.get('atol')

        order = self._method.order

        initial_step = self.select_initial_step(self.numba_model, t_start, y0, 1, order - 1, rtol,
                                                atol)  # np.min([100000000*min_step, max_step])

        strict_eval = self.method_options.get('strict_eval')
        outer_itermax = self.method_options.get('outer_itermax')

        step_integrate_ = self._method.step_func

        # figure out solve_state init
        solve_state = self._method.get_solver_state(len(y0))

        roller = self._init_roller(order)
        order_ = 0
        states = self.numba_model.get_states()
        if states.shape[0] == 0:
            for t in self.time[1:]:
                self.numba_model.func(t, states)
                if self.numba_model.is_external_data_update_needed(t):
                    self.load_external_data(t)
                self.numba_model.map_external_data(t)
                self.numba_model.historian_update(t)
        else:
            info = self._solve(self.numba_model,
                               solve_state, initial_step, order, order_, roller, strict_eval, outer_itermax, min_step,
                               max_step, step_integrate_, self.events, self.actions, self.g,
                               self.number_of_events, self.event_directions, self.run_event_action, self.timestamps,
                               self.timestamps_actions, t_start, t_end,
                               self.time, 1, atol)

            while info.status == SolveStatus.Running:
                if info.event_id == 1:
                    self.model.create_historian_df()
                    self.numba_model.historian_reinit()
                elif info.event_id == 2:
                    self.load_external_data(info.t)
                elif info.event_id == 3:
                    raise NotImplementedError

                info = self._solve(self.numba_model,
                                   info.solve_state, info.dt, order, info.order_, info.roller, strict_eval,
                                   outer_itermax,
                                   min_step, max_step, step_integrate_, self.events, self.actions, self.g,
                                   self.number_of_events, self.event_directions, self.run_event_action, self.timestamps,
                                   self.timestamps_actions, info.t, t_end,
                                   self.time, info.ix_eval, atol)
        logging.info("Solve finished")
        return self.sol, self.result_status

    def solver_step(self, t, delta_t=None):

        solve_state = self._method.get_solver_state(len(self.y0))
        t_start = t

        if delta_t is None:
            delta_t = self.delta_t

        strict_eval = self.method_options.get('strict_eval')
        outer_itermax = self.method_options.get('outer_itermax')

        max_step = self.method_options.get('max_step')
        min_step = self.method_options.get('min_step')
        rtol = self.method_options.get('rtol')
        atol = self.method_options.get('atol')
        order = self._method.order
        if self.info is not None:
            dt = self.info.dt  # internal solver step size
            order_ = self.info.order_
            roller = self.info.roller
            solve_state = self.info.solve_state

            assert self.info.t == t_start, f"solver time {self.info.t} does not match external time " \
                                           f"{t_start}"
        else:
            roller = self._init_roller(order)
            order_ = 0

            dt = self.select_initial_step(self.numba_model, t_start, self.y0, 1, order - 1, rtol,
                                          atol)

        t_end = t_start + delta_t

        time_span = np.linspace(t_start, t_end, 2)

        step_integrate_ = self._method.step_func
        info = self._solve(self.numba_model,
                           solve_state, dt, order, order_, roller, strict_eval, outer_itermax, min_step,
                           max_step, step_integrate_, self.events, self.actions, self.g,
                           self.number_of_events, self.event_directions,
                           self.run_event_action, self.timestamps, self.timestamps_actions,
                           t_start, t_end, time_span, 1, atol)

        if info.event_id == 1:
            self.model.create_historian_df()
            self.numba_model.historian_reinit()
        elif info.event_id == 2:
            self.load_external_data(t)
        elif info.event_id == 3:
            raise NotImplementedError

        self.info = info

        return t_end, self.info.t

    def register_endstep(self, __end_step):
        self.__end_step = __end_step
