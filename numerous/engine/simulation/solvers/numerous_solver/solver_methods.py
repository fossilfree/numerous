import numpy as np
from numba import njit
from scipy import linalg
import logging

from .linalg.lapack.lapack_python import lapack_solve_triangular, lapack_cholesky

class BaseMethod:
    def __init__(self):
        raise NotImplementedError
    def get_solver_state(self, *args, **kwargs):
        raise NotImplementedError


class RK45(BaseMethod):
    def __init__(self, **options):
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

        b[0, 0] = 25 / 216
        b[0, 1] = 0
        b[0, 2] = 1408 / 2565
        b[0, 3] = 2197 / 4104
        b[0, 4] = -1 / 5
        b[0, 5] = 0

        b[1, 0] = 16 / 135
        b[1, 1] = 0
        b[1, 2] = 6656 / 12825
        b[1, 3] = 28561 / 56430
        b[1, 4] = -9 / 50
        b[1, 5] = 2 / 55

        self.a = a
        self.b = b
        self.c = c

        self.max_factor = options.get('max_factor', 10)
        self.atol = options.get('atol', 1e-3)
        self.rtol = options.get('rtol', 1e-3)


        self.order = 5



        @njit
        def Rk45(nm, t, dt, y, _not_used1, _not_used2, _solve_state):




            c = _solve_state[0]
            a = _solve_state[1]
            b = _solve_state[2]
            max_factor = _solve_state[3]
            atol = _solve_state[4]
            rtol = _solve_state[5]
            step_info = 1

            converged = False
            order = 5
            tnew = t+dt

            if len(y) == 0:
                return tnew, y, True, 1, _solve_state, max_factor

            k = np.zeros((order+1, len(y)))
            k[0,:] = dt*nm.func(t, y)


            rk_sum_4 = k[0,:]*b[0,0]
            rk_sum_5 = k[0,:]*b[1,0]
            for i in range(1,order+1):
                dy = np.dot(k[:i].T, a[i,:i])
                k[i,:] = dt*nm.func(t+c[i]*dt, y+dy)

                rk_sum_4 += b[0, i] * k[i]
                rk_sum_5 += b[1, i] * k[i]

            y5 = y+rk_sum_5
            y4 = y+rk_sum_4

            res = np.linalg.norm((y5-y4)/dt)

            e_max = atol + np.max(np.abs(y5))*rtol

            if res == 0:
                delta = max_factor
            else:
                delta = (e_max/(2*res))**(1/4)
            if res <= e_max:
                converged = True
                #print(res)

            if delta < 0.1:
                factor = 0.1
            elif delta >= max_factor:
                factor = max_factor
            else:
                factor = delta


            ynew = y5

            return tnew, ynew, converged, step_info, _solve_state, factor

        self.step_func = Rk45

    def get_solver_state(self, *args, **kwargs):
        return (self.c, self.a, self.b, self.max_factor, self.atol, self.rtol)

class LevenbergMarquardt(BaseMethod):
    def __init__(self, **options):
        self.order = 5

        l_init = options.get('l', 1)
        self.l_init = l_init
        # l = options.get('l', 1)
        # Get options
        inner_itermax = options.get('inner_itermax', 4)
        S_tol = options.get('S_tol', 1e-8)
        p_tol = options.get('p_tol', 1e-8)
        grad_tol = options.get('grad_tol', 1e-8)
        jac_update_iter = options.get('jac_update_iter', 10)
        jac_update_res_change = options.get('jac_update_res_change', 10)
        update_jacobian_ = options.get('update_jacobian', True)
        abs_tol = options.get('atol', 0.001)
        rel_tol = options.get('rtol', 0.001)
        self.longer = options.get('longer', 1.2)
        self.shorter = options.get('longer', 0.8)
        lp = options.get('lp', None)
        profile = False
        if lp is not None:
            profile = True

        eps = np.finfo(1.0).eps
        newton_tol = max(10 * eps / rel_tol, min(0.03, rel_tol ** 0.5))

        s = options.get('s', 0)
        jacobian_stepsize = options.get('jacobian_stepsize', )

        af = np.array([1, 2 / 3, 6 / 11, 12 / 25, 60 / 137], dtype=np.float)
        a = np.array([[-1, 0, 0, 0, 0], [1 / 3, -4 / 3, 0, 0, 0], [-2 / 11, 9 / 11, -18 / 11, 0, 0],
                      [3 / 25, -16 / 25, 36 / 25, -48 / 25, 0],
                      [-12 / 137, 75 / 137, -200 / 137, 300 / 137, -300 / 137]], dtype=np.float)


        def comp(fun):
            if not profile:
                return njit(fun)
            else:
                return options['lp'](fun)


        @comp
        def calc_residual(r):
            Stemp = np.sum(r ** 2)
            S = Stemp / len(r)
            return S

        @comp
        def guess_init(yold, order, last_f, dt):
            _sum = np.zeros_like(yold[0, :])
            for i in range(order):
                _sum += a[order - 1][i] * yold[i, :]
            return -_sum + af[order - 1] * last_f * dt

        @comp
        def sparsejacobian(get_f, get_f_ix, __internal_state, t, y, s, dt, jacobian_stepsize):
            # Limits the number of equation calls by assuming a sparse jacobian matrix, for example in finite element
            num_eq_vars = len(y)
            jac = np.zeros((num_eq_vars, num_eq_vars))
            np.fill_diagonal(jac, 1)
            h = jacobian_stepsize

            f = get_f(t, y, __internal_state)
            y_perm = np.copy(y)

            for i in range(len(y)):
                col_m = max(0, i - s)
                col_p = min(num_eq_vars - 1, i + s)
                idx = [k for k in range(col_m, col_p + 1)]

                for j in idx:
                    y_perm[j] += h
                    f_perm = get_f_ix(t, y_perm, __internal_state, i, j)
                    y_perm[j] -= h
                    diff = (f_perm - f[i]) / h
                    jac[i][j] += -dt * diff

            return jac

        @comp
        def nonesensejac(get_f, get_f_ix, __internal_state, t, y, _, dt, jacobian_stepsize):
            jac = np.zeros((len(y), len(y)))
            np.fill_diagonal(jac, 1)
            return jac

        @comp
        def fulljacobian(get_f, get_f_ix, __internal_state, t, y, _, dt, jacobian_stepsize):
            num_eq_vars = len(y)
            jac = np.zeros((num_eq_vars, num_eq_vars))
            np.fill_diagonal(jac, 1)

            h = jacobian_stepsize

            f = get_f(t, y, __internal_state)

            for i in range(num_eq_vars):

                for j in range(num_eq_vars):
                    yperm = np.copy(y)

                    yperm[j] += h

                    # f = self.get_f_ix(t, np.array(y, dtype=float), i, j)

                    f_h = get_f(t, yperm, __internal_state)

                    # f_h = get_f_ix(t, yperm, __internal_state, i, j)

                    diff = (f_h[i] - f[i]) / h
                    # print(diff)
                    jac[j][i] += -dt * diff

            return jac

        # @comp
        @comp
        def levenberg_marquardt_inner(nm, t, dt, yinit, yold, jacT, L, order):
            # ll=l
            ll = 0
            stat = 0
            # first estimate of next step

            # TODO: give last known derivative value, instead of calculating again
            y = yinit.copy()

            _iter = 0

            scale = abs_tol + rel_tol * np.abs(y)

            x = np.zeros_like(y)
            d = np.zeros_like(x)
            rtest = np.zeros_like(y)

            x_norm_old = None

            converged = False

            while True:
                _iter += 1

                r, f = nm.get_g(t + dt, yold, y, dt, order , a, af)
                b = jacT @ r

                # x = solve_triangular(L[0] + ll * DD, -b) # deprecated
                # x = solve_triangular(L[0], -b) # From Cholesky decomposition
                # x = np.linalg.solve(L[1], -(L[0].T @ b)) # From QR decomposition - using numpy
                # x = np.linalg.solve(L[0] @ L[1], -b) # Ax = b - using Numpy
                # x = solve_qr_fun(L[0], L[1], -b) # From decomposition using own algorithms
                x = lapack_solve_triangular(L, -b, len(b))

                d += x
                y += x

                # Check if residual is decreasing
                rate = np.linalg.norm(b, ord=2)
                if rate < newton_tol:
                    converged = True
                    break

                if _iter > inner_itermax:
                    stat = -2
                    break

            return converged, rate, y, d, rtest, ll, stat, f

        @comp
        def levenberg_marquardt(nm, t, dt, y, yold, order, _solve_state):
            n = len(y)
            ynew = np.zeros_like(y)

            J = _solve_state[0]
            jacT = _solve_state[1]
            update_jacobian = _solve_state[2]
            last_f = _solve_state[3]
            L = _solve_state[4]
            updated = _solve_state[5]
            update_L = _solve_state[6]
            jac_updates = _solve_state[7]
            longer = _solve_state[8]
            shorter = _solve_state[8]

            converged = False
            y = guess_init(yold, order, last_f, dt)

            while not converged:

                # We need a smarter way to update the jacobian...
                if update_jacobian:
                    # print("update")
                    J = nm.vectorizedfulljacobian(t + dt, y, dt)

                    update_jacobian = False
                    updated = True
                    update_L = True
                    jac_updates += 1
                    # print(jac_updates, t, dt)

                if update_L:
                    jac = np.identity(n) - af[order - 1] * dt * J
                    jacT = jac.T
                    jacjacT = jac.T @ jac
                    L = lapack_cholesky(76, n, jacjacT)
                    update_L = False

                converged, S, ynew, d, r, lnew, inner_stat, f = \
                    levenberg_marquardt_inner(nm, t, dt, y, yold, jacT, L, order)

                if not converged:
                    if not updated:
                        update_jacobian = True
                    else:
                        update_L = True
                        break
                if converged:
                    updated = False
                    last_f = f

                # info = {'outer_stat': outer_stat, 'inner_stat': inner_stat, 'p_conv': p_conv, 'S': S,
                #        'grad_conv': grad_conv, 'outer_iter': outer_iter}
                # TODO: look at this dict giving issues
                # _solve_state = (jacT, L, l, converged, last_f)

            _solve_state = (J, jacT, update_jacobian, last_f, L, updated, update_L, jac_updates, longer, shorter)
            t += dt
            # print(t)
            if not converged:
                factor = shorter
            else:
                factor = longer

            return t, ynew, converged, update_jacobian, _solve_state, factor

        if profile:
            self.lp = options.get('lp')
            self.lp.add_function(levenberg_marquardt)
            self.lp.add_function(levenberg_marquardt_inner)
            # self.lp.add_function(nm.vectorizedfulljacobian)

        self.step_func = levenberg_marquardt

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
                 )

        return state
