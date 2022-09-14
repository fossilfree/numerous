import numpy as np
from numba import njit
from abc import ABC, abstractmethod
from numerous.engine.model.compiled_model import CompiledModel
from numerous.engine.simulation.solvers.numerous_solver.interface import ModelInterface, jithelper
from .linalg.lapack.lapack_python import lapack_solve_triangular, lapack_cholesky

def _njit(jit=True):
    def wrapper(fun):
        if jit:
            return njit(fun)
        else:
            return fun
    return wrapper




class BaseMethod(ABC):

    def __init__(self, numerous_solver, **options):
        self.jit_solver = numerous_solver.numba_compiled_solver

    @abstractmethod
    def get_solver_state(self, *args, **kwargs):
        pass

    @staticmethod
    def step_func(nm: CompiledModel, t: float, dt: float, y: list, yold: list, order: int, _solve_state: tuple):
        raise NotImplementedError


class RK45(BaseMethod):
    def __init__(self, numerous_solver, **options):
        super(RK45, self).__init__(numerous_solver, **options)
        __njit = _njit(self.jit_solver)

        submethod=options['submethod']

        if submethod == None:
            submethod = 'RKDP45'

        if submethod == 'RKF45':

            c = np.zeros(6)
            c[0] = 0
            c[1] = 1 / 4
            c[2] = 3 / 8
            c[3] = 12 / 13
            c[4] = 1
            c[5] = 1 / 2
            a = np.zeros((6, 5))
            b = np.zeros((2, 6))
            a[1, 0] = 1 / 4
            a[2, 0] = 3 / 32
            a[2, 1] = 9 / 32
            a[3, 0] = 1932 / 2197
            a[3, 1] = -7200 / 2197
            a[3, 2] = 7296 / 2197
            a[4, 0] = 439 / 216
            a[4, 1] = -8
            a[4, 2] = 3680 / 513
            a[4, 3] = -845 / 4104
            a[5, 0] = -8 / 27
            a[5, 1] = 2
            a[5, 2] = -3544 / 2565
            a[5, 3] = 1859 / 4104
            a[5, 4] = -11 / 40

            b[1, 0] = 16 / 135
            b[1, 1] = 0
            b[1, 2] = 6656 / 12825
            b[1, 3] = 28561 / 56430
            b[1, 4] = -9 / 50
            b[1, 5] = 2 / 55

            b[0, 0] = 25 / 216
            b[0, 1] = 0
            b[0, 2] = 1408 / 2565
            b[0, 3] = 2197 / 4104
            b[0, 4] = -1 / 5
            b[0, 5] = 0
            self.order = 4
            self.rk_steps = 5
            self.e_order = 5
        elif submethod == 'RKDP45':
            c=np.zeros(7)
            c[0] = 0
            c[1] = 1/5
            c[2] = 3/10
            c[3] = 4/5
            c[4] = 8/9
            c[5] = 1
            c[6] = 1

            a = np.zeros((6,6))
            a[1:] = [1/5, 0, 0, 0, 0 ,0]
            a[2:] = [3/40, 9/40, 0, 0, 0, 0]
            a[3:] = [44/45, -56/15, 32/9, 0, 0, 0]
            a[4:] = [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0]
            a[5:] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0]

            b = np.zeros((2,7))
            b[0:] = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
            b[1:] = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]
            self.e_order = 4
            self.order = 5
            self.rk_steps = 5

        else:
            raise Exception("incorrect submethod specified")

        self.f0 = np.ascontiguousarray(numerous_solver.f0)
        self.a = a
        self.b = b
        self.c = c

        self.error_exponent = (-1 / (self.e_order + 1))
        self.max_factor = options.get('max_factor', 10)
        self.atol = options.get('atol', 1e-3)
        self.rtol = options.get('rtol', 1e-3)

        @__njit
        def Rk45(interface, t, dt, y, _not_used1, _not_used2, _solve_state):

            c = _solve_state[0]
            a = _solve_state[1]
            b = _solve_state[2]
            max_factor = _solve_state[3]
            atol = _solve_state[4]
            rtol = _solve_state[5]
            f0 = _solve_state[6]
            rk_steps = _solve_state[7]
            order = _solve_state[8]
            error_exponent = _solve_state[9]
            last_step = _solve_state[10]
            step_info = 1

            converged = False

            tnew = t+dt

            if len(y) == 0:
                return tnew, y, True, 1, _solve_state, max_factor

            k = np.zeros((rk_steps+2, len(y)))
            k[0,:] = f0*dt

            for i in range(1,rk_steps+1):
                dy = np.dot(k[:i].T, a[i,:i])
                interface.set_states(y + dy)
                k[i, :] = dt * interface.get_deriv(t + c[i] * dt)

            ynew = y + np.dot(k[0:order+2].T, b[0,:])
            #fnew = nm.func(tnew, ynew)  # can possibly save one call here...
            interface.set_states(ynew)
            fnew = interface.get_deriv(tnew)
            k[-1, :] = dt*fnew

            ye = y + np.dot(k[0:order+2].T, b[1,:])
            scale = atol + np.maximum(np.abs(y), np.abs(ynew)) * rtol

            e = (ynew-ye)

            e_norm = np.linalg.norm(e/scale)/ (len(e)**0.5)

            if e_norm < 1:
                if e_norm == 0:
                    factor = max_factor
                else:
                    factor = min(max_factor,
                                 0.95 * e_norm ** error_exponent)

                if not last_step:
                    factor = min(1, factor)

                f0=fnew
                converged = True
            else:
                factor = max(0.2,
                             0.95 * e_norm ** error_exponent)

            _new_solve_state = (c, a, b, max_factor, atol, rtol, np.ascontiguousarray(f0), rk_steps, order, error_exponent, converged)

            return tnew, ynew, converged, step_info, _new_solve_state, factor

        self.step_func = Rk45

    def get_solver_state(self, *args, **kwargs):
        return (self.c, self.a, self.b, self.max_factor, self.atol, self.rtol, self.f0, self.rk_steps, self.order,
                self.error_exponent, True)


class BDF(BaseMethod):
    def __init__(self, numerous_solver, **options):
        super(BDF, self).__init__(numerous_solver, **options)
        numba_compiled_solver = numerous_solver.numba_compiled_solver

        __njit = _njit(numba_compiled_solver)

        self.max_order = 5

        l_init = options.get('l', 1)
        self.l_init = l_init
        # Get options
        inner_itermax = 100#options.get('inner_itermax', 4)
        abs_tol = options.get('atol', 0.001)
        rel_tol = options.get('rtol', 0.001)
        self.longer = options.get('longer', 1.2)
        self.shorter = options.get('shorter', 0.8)
        jacobian = numerous_solver.interface.model.get_jacobian
        h = options.get('jacobian_stepsize', 1e-8)
        y0 = numerous_solver.y0
        initial_step = numerous_solver.select_initial_step(numerous_solver.interface, numerous_solver.time_[0], y0, 1,
                                                           self.max_order - 1, rel_tol,
                                                           abs_tol)

        eps = np.finfo(1.0).eps
        newton_tol = max(10 * eps / rel_tol, min(0.03, rel_tol ** 0.5))

        af = np.array([1, 2 / 3, 6 / 11, 12 / 25, 60 / 137], dtype=np.float)
        a = np.array([[-1, 0, 0, 0, 0], [1 / 3, -4 / 3, 0, 0, 0], [-2 / 11, 9 / 11, -18 / 11, 0, 0],
                      [3 / 25, -16 / 25, 36 / 25, -48 / 25, 0],
                      [-12 / 137, 75 / 137, -200 / 137, 300 / 137, -300 / 137]], dtype=np.float)

        @__njit
        def update_D(D, order, factor):
            R = calculate_R(order, factor)
            U = calculate_R(order, 1)
            RU = R.dot(U)
            D[:order + 1] = np.dot(RU.T, D[:order +1])
            return D

        @__njit
        def calculate_R(order, factor):
            I = np.arange(1, order + 1)[:, None]
            J = np.arange(1, order + 1)
            M = np.zeros((order+1, order+1))
            M[1:, 1:] = (I - 1 - factor * J) / I
            M[0] = 1
            return np.cumprod(M, axis=0)



        gamma = np.hstack((0, np.cumsum(1/np.arange(1, self.max_order+1))))
        kappa = np.array([0, -1.85, -1/9, -0.0823, -0.0415, 0])
        alpha = (1-kappa)*gamma
        D = np.empty(self.max_order + 3, len(y0))
        D[0] = y0
        D[1] = numerous_solver.f0 * initial_step

        @__njit
        def calc_residual(r):
            Stemp = np.sum(r ** 2)
            S = Stemp / len(r)
            return S

        @__njit
        def guess_init(yold, order, last_f, dt):
            _sum = np.zeros_like(yold[0, :])
            for i in range(order):
                _sum += a[order - 1][i] * yold[i, :]
            return -_sum + af[order - 1] * last_f * dt

        @__njit
        def get_residual(interface, t, yold, y, dt, order, a, af):
            interface.set_states(y)
            f = interface.get_deriv(t)
            _sum = np.zeros_like(y)
            for i in range(order):
                _sum = _sum + a[order - 1][i] * yold[i, :]
            g = y + _sum - af[order - 1] * dt * f
            return np.ascontiguousarray(g), np.ascontiguousarray(f)

        @__njit
        def bdf_inner(interface, t, y_init, psi, c, scale, L):

            stat = 0
            # first estimate of next step

            # TODO: give last known derivative value, instead of calculating again
            y = y_init.copy()

            _iter = 0

            d = np.zeros_like(y)
            converged = False
            dy_norm_old = None
            rate = None

            for k in range(inner_itermax):
                _iter += 1
                interface.set_states(y)
                f = interface.get_deriv(t)

                b = c * f - psi - d

                dy = lapack_solve_triangular(L, -b, len(b))
                dy_norm = np.linalg.norm(dy / scale)
                if not dy_norm_old:
                    rate = None
                else:
                    rate = dy_norm / dy_norm_old

                if rate is not None and (rate >= 1 or rate ** (inner_itermax - k) / (1-rate) * dy_norm > newton_tol):
                    stat = -1
                    break

                d += dy
                y += dy

                if dy_norm == 0 or (rate is not None and rate / (1-rate) * dy_norm < newton_tol):
                    converged = True
                    break

                dy_norm_old = dy_norm


            return converged, rate, y, d, stat

        @__njit
        def bdf(interface, t, dt, y, _, __, _solve_state):
            n = len(y)
            ynew = np.zeros_like(y)

            J = _solve_state[0]
            jacT = _solve_state[1]
            update_jacobian = False#_solve_state[2]
            last_f = _solve_state[3]
            L = _solve_state[4]
            updated = _solve_state[5]
            update_L = _solve_state[6]
            jac_updates = _solve_state[7]
            longer = _solve_state[8]
            shorter = _solve_state[9]
            D = _solve_state[10]
            dt_last = _solve_state[11]
            order = _solve_state[12]

            converged = False

            if dt != dt_last:
                update_D(D, order, dt/dt_last)

            psi = 1/alpha[order] * np.sum(gamma[1:order+1] * D[1:order+1])

            y_init = np.sum(D[:order + 1], axis=0)  # euler integration
            scale = abs_tol + rel_tol * np.abs(y_init)
            c = dt/(alpha[order])
            t += dt

            while not converged:

                # We need a smarter way to update the jacobian...
                if update_jacobian:
                    interface.set_states(y)
                    J = jacobian(t, h)

                    update_jacobian = False
                    updated = True
                    update_L = True
                    jac_updates += 1

                if update_L:
                    jac = np.identity(n) - c * J
                    L = lapack_cholesky(76, n, jac)
                    update_L = False

                converged, S, ynew, d, lnew, inner_stat, f = \
                    bdf_inner(interface, t, y_init, psi, c, scale, L)

                interface.set_states(ynew)

                if not converged:
                    if not updated:
                        update_jacobian = True
                    else:
                        update_L = True
                        break

                if converged:
                    updated = False
                    last_f = f

            _solve_state = (J, jacT, update_jacobian, last_f, L, updated, update_L, jac_updates, longer, shorter, D, dt)

            # print(t)
            if not converged:
                factor = shorter
            else:
                factor = longer

            return t, ynew, converged, update_jacobian, _solve_state, factor

        self.step_func = bdf


    def get_solver_state(self, n):

        state = (np.ascontiguousarray(np.zeros((n, n))),
                 np.ascontiguousarray(np.zeros((n, n))),
                 True,
                 np.ascontiguousarray(np.zeros(n)),
                 np.ascontiguousarray(np.zeros((n, n))),
                 False,
                 True,
                 0,
                 self.longer,
                 self.shorter,
                 self.D,
                 self.initial_step
                 )

        return state


class Euler(BaseMethod):
    def __init__(self, numerous_solver, **options):

        self.order = 1
        self.max_factor = options.get('max_factor', 10)
        self.atol = options.get('atol', 1e-3)
        self.rtol = options.get('rtol', 1e-3)

        numba_compiled_solver = numerous_solver.numba_compiled_solver

        def njit_(fun):
            if numba_compiled_solver:
                return njit(fun)
            else:
                # return options['lp'](fun)
                return fun

        @njit_
        def euler(nm, t, dt, y, _not_used1, _not_used2, _solve_state):

            step_info = 1

            tnew = t + dt

            if len(y) == 0:
                return tnew, y, True, step_info, _solve_state, 1e20

            fnew = nm.func(t, y)

            ynew = y + fnew * dt
            # TODO figure out if this call can be avoided
            nm.func(tnew, ynew)
            return tnew, ynew, True, step_info, _solve_state, 1e20

        self.step_func = euler

    def get_solver_state(self, *args, **kwargs):
        return (0.0, 0.0, 0.0, self.max_factor, self.atol, self.rtol, 0.0, 1, self.order,
                1, True)
