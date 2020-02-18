import numpy as np
import copy
from datetime import datetime


from numerous.engine.simulation import Simulation

class SteadyStateSimulation(Simulation):
    def __init__(self, model,
                 start_datetime=datetime.now(), S_tol=1e-8, p_tol=1e-8, grad_tol=1e-8,
                 jacobian_stepsize=1e-12, sparse_elements=1, outer_itermax=1000, inner_itermax=1000,
                 jac_update_iter = 10, jac_update_res_change=0.5,
                 jac_method=3, **kwargs):
        super(SteadyStateSimulation, self).__init__(model=model, num=outer_itermax+1, num_inner=1,
                          start_datetime=start_datetime, **kwargs)
        """
        :param fun: must return the function vector to be minimized using the Levenberg-Marquardt algorithm
        (Non linear least squares)
        """


        self.f = []
        #self.fun = fun
        self.variables = {}
        self.parameters = {}
        #self.varsdict = {}
        self.vars = []
        self.l = 1
        self.conv = None
        self.iter = 0
        self.num_eq_vars = None
        self.sys_dict = None
        self.outputs = {}
        self.S = 0
        #self.conv_tol = conv_tol
        self.S_tol = S_tol
        self.p_tol = p_tol
        self.grad_tol = grad_tol
        self.outer_itermax = outer_itermax
        self.inner_itermax = inner_itermax
        self.jacobian_stepsize = jacobian_stepsize
        self.sparse_elements = sparse_elements
        self.jac_update_iter = jac_update_iter
        self.jac_update_res_change = jac_update_res_change
        self.jacT = None
        self.jacjacT = None

        if jac_method == 1:
            self.jacobian_func = self.jacobian
        elif jac_method == 2:
            self.jacobian_func = self.sparsejacobian
        elif jac_method == 3:
            self.jacobian_func = self.experimentaljacobian
        else:
            raise Exception('Unknown choice of jacobian method')


        #print(num_eq, self.num_vars)


    def set_vars(self,_vars):
        for var in _vars:
            self.vars[var] = _vars[var]

    def set_inputs(self, inputs):
        for _input in inputs:
            self.inputs[_input] = inputs[_input]

    def set_vars(self, vars_):
        self.vars = vars_
        variables = self.variables
        for j in range(0,len(vars_)):
            var = vars_[j]['name']
            val = vars_[j]['val']
            variables[var] = val
        self.variables = variables

    def jacobian(self, y):

        jac = np.zeros((self.num_eq_vars, self.num_eq_vars))

        f, _ = self.get_f(y)

        h = 1e-12

        for i in range(self.num_eq_vars):

            for j in range(self.num_eq_vars):
                yperm = list(y)

                yperm[j] += h

                f_h, _ = self.get_f(yperm)

                diff = (f_h[i] - f[i]) / h

                jac[i][j] = diff

        return jac

    def sparsejacobian(self, y):

        #Limits the number of equation calls by assuming a sparse jacobian matrix, for example in finite element

        jac = np.zeros((self.num_eq_vars, self.num_eq_vars))
        f_h_mat = np.zeros((self.num_eq_vars, self.num_eq_vars))
        h = self.jacobian_stepsize

        f, _ = self.get_f(y)

        s = self.sparse_elements # elements
        yperm = list(y)
        for i in range(self.num_eq_vars):
            yperm[i] += h
            f_h, _ = self.get_f(yperm)
            yperm[i] -= h
            f_h_mat[:,i] = f_h

        for i in range(self.num_eq_vars):
            col_m = max(0, i-s)
            col_p = min(self.num_eq_vars-1, i+s)
            idx = np.linspace(col_m, col_p, col_p-col_m+1, dtype=int)

            for j in idx:

                diff = (f_h_mat[i,j] - f[i]) / h

                jac[i][j] = diff
            pass

        return jac

    def experimentaljacobian(self, y):

        #Limits the number of equation calls by assuming a sparse jacobian matrix, for example in finite element

        jac = np.zeros((self.num_eq_vars, self.num_eq_vars))
        h = self.jacobian_stepsize

        f, _ = self.get_f(y)

        s = self.sparse_elements # elements

        for i in range(self.num_eq_vars):
            col_m = max(0, i - s)
            col_p = min(self.num_eq_vars - 1, i + s)
            idx = np.linspace(col_m, col_p, col_p - col_m + 1, dtype=int)

            for j in idx:
                y_j = y[j]
                f_h = self.__jacfunc(y_j, h, i, j)
                diff = (f_h - f[i]) / h
                jac[i][j] = diff

        return jac

    def calc_residual(self, r):
        Stemp = np.sum(np.array(r) ** 2)
        S = Stemp/len(r)
        return S

    def inner(self, yin, update_jacobian=True):
        l=self.l
        y = list(yin)
        r, stat = self.get_f(y)
        S = self.calc_residual(r)

        if update_jacobian:

            jac = self.jacobian_func(y)
            #jac_test = self.jacobian(y)
            jacT = jac.T
            jacjacT = np.matmul(jac, jacT)
            self.jacT = jacT
            self.jacjacT = jacjacT

        else:
            jacT = self.jacT
            jacjacT = self.jacjacT


        D = np.zeros((self.num_eq_vars, self.num_eq_vars))
        diagmat = np.zeros((self.num_eq_vars, self.num_eq_vars))
        np.fill_diagonal(D, 1)
        jacjacT_diag = np.diag(jacjacT)
        np.fill_diagonal(diagmat, jacjacT_diag)
        DD = np.maximum(diagmat, D)

        b = np.matmul(jacT, r)
        _iter = 0
        itermax = self.inner_itermax

        ynew = y
        Stest = S
        x=None
        rtest=None


        while stat == 0:
            _iter += 1

            #amat = jacjacT+D*l
            amat = jacjacT + l*DD

            x = np.linalg.solve(amat, -b)

            xtest = ynew + x

            rtest, stat = self.get_f(xtest)

            Stest = self.calc_residual(rtest)
            if Stest > S:
                l = l*2
            else:
                l = l/2
                self.l = l
                ynew = tuple(xtest)
                #print(l, S, Stest)
                stat = 1
                break
                #stat = 1
            if _iter > itermax:
                stat = -2
                break

        return stat, Stest, ynew, x, rtest


    def inner_scaled(self, yin, update_jacobian=True):
        l=self.l
        y = list(yin)
        r, stat = self.get_f(y)
        S = self.calc_residual(r)

        s = np.std(y)
        m = np.mean(y)

        if update_jacobian:

            jac = self.jacobian_func(y)
            #jac_test = self.jacobian(y)
            jacT = jac.T
            jacjacT = np.matmul(jac, jacT)
            self.jacT = jacT
            self.jacjacT = jacjacT

        else:
            jacT = self.jacT
            jacjacT = self.jacjacT


        jacT = jacT * s
        jacjacT = jacjacT * (s**2)

        D = np.zeros((self.num_eq_vars, self.num_eq_vars))
        diagmat = np.zeros((self.num_eq_vars, self.num_eq_vars))
        np.fill_diagonal(D, 1)
        jacjacT_diag = np.diag(jacjacT)
        np.fill_diagonal(diagmat, jacjacT_diag)
        DD = np.maximum(diagmat, D)

        b = np.matmul(jacT, r)
        _iter = 0
        itermax = self.inner_itermax

        ynew = (y-m)/s
        Stest = S
        x=None
        rtest=None


        while stat == 0:
            _iter += 1

            #amat = jacjacT+D*l
            amat = jacjacT + l*DD

            x = np.linalg.solve(amat, -b)

            xtest = ynew + x

            rtest, stat = self.get_f(s*xtest+m)

            Stest = self.calc_residual(rtest)
            if Stest > S:
                l = l*2
            else:
                l = l/2
                self.l = l
                ynew = xtest
                #print(l, S, Stest)
                stat = 1
                break
                #stat = 1
            if _iter > itermax:
                stat = -2
                break

        return stat, Stest, tuple(s*ynew+m), x, rtest

    # return all values of equation states (dy/dt = f(t,y))
    def get_f(self, y):
        return self.__func(0, list(y))



    def solve(self):
        Sold = None
        iter = 0
        S_tol = self.S_tol
        p_tol = self.p_tol
        grad_tol = self.grad_tol
        outer_stat = 0
        self.l = 1
        jac_update_iter = self.jac_update_iter
        jac_update_res_change = self.jac_update_res_change
        y = tuple(self.y0)
        itermax = self.outer_itermax

        self.num_eq_vars = len(y)

        self._Simulation__init_step()
        update_jacobian = True
        lastupdate = 0
        while outer_stat == 0:
            inner_stat, S, ynew, x, r = self.inner_scaled(y, update_jacobian)

            test, _ = self.get_f(ynew)
            #print('inner loop out: ' + str(self.calc_residual(test)))

            iter += 1

            if (iter > itermax) or (inner_stat < 0):
                outer_stat = -1
                break

            if Sold is None:
                conv = 1
            else:
                conv = np.abs((S-Sold)/(Sold+1e-12))

            # the 3 convergence criteria
            p_conv = np.max(np.abs(x / np.array(ynew)))
            grad_conv = np.linalg.norm(np.matmul(self.jacT,r),1)

            if S < S_tol:
                outer_stat = 1
                break
            if p_conv < p_tol:
                outer_stat = 2
                break
            if grad_conv < grad_tol:
                outer_stat = 3
                break

            # should we update the Jacobian?
            update_jacobian = False
            if (conv >= jac_update_res_change) or (lastupdate > jac_update_iter):
                update_jacobian = True
                lastupdate = 0
            lastupdate+=1

            print('Solver: iteration ' + str(iter) + ' jac update ' + str(update_jacobian) + ' Residual ' + str(S) + ' parameter conv ' + str(p_conv) \
                  + ' gradient conv ' + str(grad_conv))
            y = ynew



            self._Simulation__end_step(y, iter)
            #test2, _ = self.get_f(y)
            #print('inner loop out (2): ' + str(self.calc_residual(test2)))


            Sold = S

        list(map(lambda x: x.finalize(), self.model.callbacks))
        if outer_stat < 0:
            print('Convergence failed. Status code ' + str(outer_stat) + ' (outer) '+ str(inner_stat) + ' (inner)')
        else:
            print('Solver converged. Status code ' + str(outer_stat) )
            print('Final: iteration ' + str(iter) + ' Residual ' + str(S) + ' parameter conv ' + str(p_conv) \
                  + ' gradient conv ' + str(grad_conv))

        self.p_conv = p_conv
        self.grad_conv = grad_conv
        self.residual = S
        self.inner_stat = inner_stat
        self.outer_stat = outer_stat
        self.iter = iter


        self.t_scope.update_model_from_scope(self.model)

    def __func(self, _t, y):
        # the following is done to avoid update race condition
        derivatives0 = tuple([0 for _ in y])
        iter=0
        itermax = len(y)*10
        status = 0
        while True:

            self.t_scope.update_states(y)
            list(map(lambda x: x.set_time(_t), self.t_scope.scope_dict.values()))
            self.info["Number of Equation Calls"] = self.info["Number of Equation Calls"] + 1

            for key, eq in self.model.equation_dict.items():
                scope = self.t_scope.scope_dict[key]
                for eq_method in eq:
                    eq_method(scope)
            result = self.t_scope.get_derivatives()
            derivatives = tuple([x.get_value() for x in result])
            diff = ((np.array(derivatives) - np.array(derivatives0)) ** 2).sum()
            if diff < 1e-12:
                break
            derivatives0 = derivatives
            iter+=1
            if iter > itermax:
                status = -1
                break
        return list(derivatives), status

    # used to permute equation "i" with state value "y_j", being the "j"th element in the state vector
    def __jacfunc(self, y_j, h, i_idx, j_idx):
        # the following is done to avoid update race condition
        derivatives0 = 0
        iter = 0
        itermax = 10
        status = 0
        while True:
            self.t_scope.update_states_idx(y_j+h, j_idx) # push permuted "y_j+h" into scope belonging to that state

            scope_variable = self.t_scope.resList[j_idx]
            # Update all equations in this scope variable, and it's linked scope variables
            self.__update_eq(scope_variable, 1)

            result = self.t_scope.get_derivatives_idx(i_idx)

            derivatives = result.get_value()

            self.t_scope.update_states_idx(y_j, j_idx)  # return state "y_j" into scope belonging to that state
            self.__update_eq(scope_variable, 1)

            diff = ((np.array(derivatives)- np.array(derivatives0)) **2).sum()
            if diff < 1e-12:
                break
            derivatives0 = derivatives
            iter+=1

            if iter > itermax:
                status = -1
                break

        return derivatives

    # updates the parent scopes for current and chain-mapped scopes
    def __update_eq(self, scope_variable, depth):
        scope = self.t_scope.scope_dict[scope_variable.parent_scope_id]
        for eq_method in scope_variable.bound_equation_methods:
            eq_method(scope)

        # recursive calls
        for mapped_var in scope_variable.mapping:
            self.__update_eq(mapped_var,depth+1)
        for mapped_var in scope_variable.sum_mapping:
            self.__update_eq(mapped_var,depth+1)







"""
t_e = [-10,-15, -20]

for _t_e in t_e:

    calc = Calc(t_e=_t_e, t_c=35, dt_e=30, t_eco=10, p_dot=10, dt_eco=4.9, eta=0.6612, m1_dot=412.4473, m_dot=477.5939)
    a=NRSolver(fun=calc)
    a.solve()

    print(a.vars, a.get_f(a.vars), a.conv, a.stat, a.iter)

"""
