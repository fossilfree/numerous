import numpy as np
from numba import njit
from scipy import linalg
import logging

from .linalg.lapack.lapack_python import lapack_solve_triangular, lapack_cholesky


class LevenbergMarquardt:
    def __init__(self, **options):

        itermax = options.get('itermax',1000)
        l_init = options.get('l', 1)
        self.l_init = l_init
        #l = options.get('l', 1)
        # Get options
        inner_itermax = options.get('inner_itermax', 10)
        outer_itermax = options.get('outer_itermax', 2)
        S_tol = options.get('S_tol', 1e-8)
        p_tol = options.get('p_tol', 1e-8)
        grad_tol = options.get('grad_tol', 1e-8)
        jac_update_iter = options.get('jac_update_iter', 10)
        jac_update_res_change = options.get('jac_update_res_change', 10)
        update_jacobian_ = options.get('update_jacobian', True)
        abs_tol = options.get('atol', 0.001)
        rel_tol = options.get('rtol', 0.001)
        profile = options.get('profile', False)

        s = options.get('s', 0)
        jacobian_stepsize = options.get('jacobian_stepsize', )

        def comp(fun):
            if profile:
                return options['lp'](fun)
            else:
                return fun

        @njit
        def calc_residual(r):
            Stemp = np.sum(r**2)
            S = Stemp / len(r)
            return S

        @njit
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
                    f_perm = get_f_ix(t, y_perm, __internal_state,  i, j)
                    y_perm[j] -= h
                    diff = (f_perm - f[i]) / h
                    jac[i][j] += -dt * diff

            return jac

        @njit
        def nonesensejac(get_f, get_f_ix, __internal_state, t, y, _, dt, jacobian_stepsize):
            jac = np.zeros((len(y), len(y)))
            np.fill_diagonal(jac, 1)
            return jac

        @njit
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

                    #f_h = get_f_ix(t, yperm, __internal_state, i, j)

                    diff = (f_h[i] - f[i]) / h
                    #print(diff)
                    jac[j][i] += -dt * diff

            return jac

        #@comp
        @njit
        def levenberg_marquardt_inner(nm,t, dt, ynew, yold, jacT, L, l, f_init):
            #ll=l
            ll = 0
            stat = 0
            # first estimate of next step

            #TODO: give last known derivative value, instead of calculating again


            y = ynew.copy() + dt * f_init
            #r, _ = get_g(__func, __internal_state, t, yold, y, dt)

            Sold = np.inf#calc_residual(r)

            num_eq_vars = len(y)
            D = np.zeros((num_eq_vars, num_eq_vars))
            np.fill_diagonal(D, 1.0)

            DD = D
            #b = jacT @ r
            #b = r
            _iter = 0


            scale = abs_tol + rel_tol * np.abs(y)
            #Stest = S
            #x = None
            #rtest = None

            x = np.zeros_like(y)
            d = np.zeros_like(x)
            rtest = np.zeros_like(y)

            eps = np.finfo(1.0).eps
            newton_tol = max(10 * eps / rel_tol, min(0.03, rel_tol ** 0.5))
            x_norm_old = -1

            f = np.zeros_like(y)
            converged = False

            while stat == 0:
                _iter += 1

                r, f = nm.get_g(t, yold, y, dt)
                Stest = calc_residual(r)
                b = jacT @ r

                #x = solve_triangular(L[0] + ll * DD, -b) # deprecated
                #x = solve_triangular(L[0], -b) # From Cholesky decomposition
                #x = np.linalg.solve(L[1], -(L[0].T @ b)) # From QR decomposition - using numpy
                #x = np.linalg.solve(L[0] @ L[1], -b) # Ax = b - using Numpy
                #x = solve_qr_fun(L[0], L[1], -b) # From decomposition using own algorithms
                x = lapack_solve_triangular(L, -b, len(b))

                d += x
                y += x

                # Check if residual is decreasing
                if Stest > Sold:
                    ll = ll * 2
                else:
                    ll = ll / 2

                x_norm = np.linalg.norm(x / scale) / (len(x) ** 0.5)
                if x_norm_old > 0:
                    rate = x_norm / x_norm_old
                    converged = rate / (1 - rate) * x_norm < newton_tol
                x_norm_old = x_norm
                if converged:
                    stat = 1

                if _iter > itermax:
                    stat = -2
                Sold = Stest

            return converged, Stest, y, d, rtest, ll, stat, f

        #@comp
        @njit
        def levenberg_marquardt(nm, t, dt, y, _solve_state):
            yold = np.copy(y)
            n = len(y)
            ynew = np.zeros_like(yold)
            outer_stat = 0
            outer_iter = 0

            update_jacobian = False
            jacT = _solve_state[0]
            l = _solve_state[1]
            converged = _solve_state[2]
            last_f = _solve_state[3]
            L = _solve_state[4]
            if not converged:
                update_jacobian = True


            converged = False

            while not converged:

                if update_jacobian:
                    jac = nm.vectorizedfulljacobian(t, y, dt)
                    jacT = jac.T
                    jacjacT = jac.T @ jac
                    #(L) = np.linalg.cholesky(jacjacT) # Use paranthesis to make into tuple
                    #L = np.linalg.qr(jacjacT) # returns Q,R as tuple L
                    L = lapack_cholesky(76, n, jacjacT)


                converged, S, ynew, d, r, lnew, inner_stat, last_f = \
                    levenberg_marquardt_inner(nm,t, dt, y, yold, jacT, L,l, last_f)
                l = lnew

                if not converged:
                    update_jacobian = True

                outer_iter += 1
                #print(outer_iter)
                if (outer_iter > outer_itermax):
                    outer_stat = -1
                    break



            #info = {'outer_stat': outer_stat, 'inner_stat': inner_stat, 'p_conv': p_conv, 'S': S,
            #        'grad_conv': grad_conv, 'outer_iter': outer_iter}
            # TODO: look at this dict giving issues
            #_solve_state = (jacT, L, l, converged, last_f)
            _solve_state = (jacT, l, converged, last_f, L)
            #_solve_state[0] = jacT
            #_solve_state[1] = L
            #_solve_state[2] = l
            #_solve_state[3] = converged
            #_solve_state[4] = last_f
            #info = {'a': 48}
            t += dt
            return t, ynew, converged, outer_stat, _solve_state#, l

        self.step_func = levenberg_marquardt

    def get_solver_state(self, n):
        #state = (np.ascontiguousarray(np.zeros((n, n))), np.ascontiguousarray(np.zeros((n, n))),
        #         self.l_init, False, np.ascontiguousarray(np.zeros(n)))
        state = (np.ascontiguousarray(np.zeros((n,n))),
                 self.l_init, False, np.ascontiguousarray(np.zeros(n)), np.ascontiguousarray(np.zeros((n,n)))
                 )
        return state


