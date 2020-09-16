from numerous.engine.simulation.solvers.base_solver import BaseSolver
from .solver_methods import *
import time

class Numerous_solver(BaseSolver):

    def __init__(self, time, delta_t,  numba_model, num_inner, max_event_steps, y0, **kwargs):
        super().__init__()
        self.time = time
        self.num_inner = num_inner
        self.delta_t = delta_t
        self.numba_model = numba_model
        self.diff_function = numba_model.func

        self.f0 = self.diff_function(time[0], y0)
        self.max_event_steps = max_event_steps
        self.options = kwargs

        eps = np.finfo(1.0).eps
        odesolver_options = {'longer': kwargs.get('longer', 1.2), 'shorter': kwargs.get('shorter', 0.8),
                             'min_step': kwargs.get('min_step', 10*eps), 'strict_eval': True,
                             'max_step': kwargs.get('max_step', np.inf), 'first_step': kwargs.get('first_step', None),
                             'atol': kwargs.get('atol', 1e-6), 'rtol': kwargs.get('rtol', 1e-3),
                             'outer_itermax': kwargs.get('outer_itermax', 20),
                             'submethod': kwargs.get('submethod', None),
                             }


        self.method_options = odesolver_options
        try:
            self.method = eval(kwargs.get('method', 'RK45'))
            self._method = self.method(self, **self.method_options)
            assert issubclass(self.method, BaseMethod), f"{self.method} is not a BaseMethod"
        except Exception as e:
            raise e

        self.y0 = y0


        # Generate the solver
        self._non_compiled_solve = self.generate_solver()
        self._solve = self.compile_solver()
        #self._solve = self.generate_solver()

    def generate_solver(self):
        @njit

        def _solve(numba_model, _solve_state, initial_step, order, strict_eval, outer_itermax,
                   min_step, max_step, step_integrate_,
                   t0=0.0, t_end=1000.0, t_eval=np.linspace(0.0, 1000.0, 100)):
            # Init t to t0
            t = t0
            dt = initial_step
            y = numba_model.get_states()
            if y.shape[0] == 0:
                for t in t_eval[1:]:
                    numba_model.func(t, y)
                    numba_model.historian_update(t)
                return  {'step_info': 1}
            t_start = t
            t_previous = 0
            y_previous = np.copy(y)

            order_ = 0
            len_y = numba_model.get_states().shape[0]
            n = order + 2
            rb0 = np.zeros((n, len(y)))

            roller = (n, np.zeros(n), rb0)

            roller_ix = -1

            def add_ring_buffer(t_, y_, rb, o):
                if o == order:
                    y_temp = rb[2][:, :]
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
            ix_eval = 1
            te_array[1] = t_eval[ix_eval] + dt if strict_eval else np.inf
            t_next_eval = t_eval[ix_eval]

            outer_iter = outer_itermax


            terminate = False
            step_converged = True
            dt_last = -1
            j_i = 0
            progress_c = t_eval.shape[0]
            while not terminate:
                # updated events time estimates
                # # time acceleration
                if not step_converged:

                    if min_step > dt:
                        raise ValueError('dt shortened below min_dt')
                    decrease = True
                    te_array[0] = t_previous+dt
                else:
                    dt = min(max_step, dt)
                    decrease = False

                    te_array[0] = t + dt

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
                    if t_next_eval <= (t + 10*np.finfo(1.0).eps):
                        j_i += 1
                        p_size = 100
                        x = int(p_size * j_i / progress_c)
                        print(t)
                        numba_model.historian_update(t)
                        #numba_model.run_callbacks_with_updates(t)
                        if strict_eval:
                            te_array[1] = t_next_eval = t_eval[ix_eval + 1] if ix_eval + 1 < len(t_eval) else t_eval[-1]
                        else:
                            t_next_eval = t_eval[ix_eval + 1] if ix_eval + 1 < len(t_eval) else t_eval[-1]
                        ix_eval += 1
                        dt = initial_step
                        te_array[0] = t+dt

                    t_start = t
                    t_new_test = np.min(te_array)
                    if t >= t_end:
                        break

                    order_ = add_ring_buffer(t, y, roller, order_)

                dt_ = min([t_next_eval-t_start, t_new_test-t_start])

                #dt_ = t_next_eval - t_start

                    # solve from start to new test by calling the step function

                t, y, step_converged, step_info, _solve_state, factor = step_integrate_(numba_model,
                                                                                t_start,
                                                                                dt_, y,
                                                                                get_order_y(roller, order_), order_,
                                                                                _solve_state)

                dt *= factor

                if step_converged:
                    y_previous = y
                    t_previous = t
                    #numba_model.func(t, y)
                    #print(t)

            info = {'step_info': step_info}
            # Return the part of the history actually filled in
            # print(history[1:,0])
            return info

        return _solve

    def compile_solver(self):

        print("Compiling Numerous Solver")
        generation_start = time.time()

        argtypes = []

        max_step = self.method_options.get('max_step')
        min_step = self.method_options.get('min_step')



        strict_eval = self.method_options.get('strict_eval', True)
        outer_itermax = self.method_options.get('outer_itermax', 20)



        order = self._method.order
        initial_step = min_step

        step_integrate_ = self._method.step_func

        args = (self.numba_model,
                self._method.get_solver_state(len(self.y0)),initial_step,
                order, strict_eval, outer_itermax, min_step,
                max_step, step_integrate_,
                self.time[0],
                self.time[-1],
                self.time)
        for a in args:
            argtypes.append(self._non_compiled_solve.typeof_pyval(a))
        # Return the solver function

        _solve= self._non_compiled_solve.compile(tuple(argtypes))


        generation_finish = time.time()
        print("Compiling time: ", generation_finish - generation_start)

        return _solve

    def set_state_vector(self, states_as_vector):
        self.y0 = states_as_vector

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

        # try:
        t_start = self.time[0]
        t_end = self.time[-1]

        # Call the solver
        from copy import deepcopy
        y0 = deepcopy(self.y0)

        state = deepcopy(self.y0)



        # Set options

        max_step = self.method_options.get('max_step')
        min_step = self.method_options.get('min_step')
        rtol = self.method_options.get('rtol')
        atol = self.method_options.get('atol')

        order = self._method.order

        initial_step = self.select_initial_step(self.numba_model, t_start,  y0, 1, order-1, rtol, atol )  # np.min([100000000*min_step, max_step])

        strict_eval = self.method_options.get('strict_eval')
        outer_itermax = self.method_options.get('outer_itermax')


        step_integrate_ = self._method.step_func



        # figure out solve_state init
        solve_state = self._method.get_solver_state(len(y0))
        info = self._solve(self.numba_model,
                           solve_state, initial_step, order, strict_eval, outer_itermax, min_step,
                           max_step, step_integrate_,
                           t_start, t_end, self.time)
        print("finished")
        # except Exception as e:
        #     print(e)
        #     raise e
        # finally:
        return self.sol, self.result_status

    def register_endstep(self, __end_step):
        self.__end_step = __end_step
