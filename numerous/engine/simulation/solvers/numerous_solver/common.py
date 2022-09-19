import numpy as np
import numba as nb
from numba.experimental import jitclass

from .interface import ModelInterface

EPS = np.finfo(1.0).eps

NUM_JAC_DIFF_REJECT = EPS ** 0.875
NUM_JAC_DIFF_SMALL = EPS ** 0.75
NUM_JAC_DIFF_BIG = EPS ** 0.25
NUM_JAC_MIN_FACTOR = 1e3 * EPS
NUM_JAC_FACTOR_INCREASE = 10
NUM_JAC_FACTOR_DECREASE = 0.1


def jithelper(model_class: callable, jit=True):
    def wrapper(*args, **kwargs):
        model = model_class(*args, **kwargs)
        if not jit:
            return model

        spec = []
        for v in model.__dict__:
            nbtype = nb.typeof(getattr(model, v))
            if type(nbtype) == nb.types.Array:  # Array must be type 'A' -
                # by default the nb.typeof evaluates them to type 'C'
                spec.append((v, nb.types.Array(nbtype.dtype, nbtype.ndim, 'A')))
            else:
                spec.append((v, nbtype))

        @jitclass(spec=spec)
        class Wrapper(model_class):
            pass

        return Wrapper(*args, **kwargs)
    return wrapper


def norm(x):
    return np.linalg.norm(x) / len(x) ** 0.5


def select_initial_step(interface: ModelInterface, t0, y0, direction, order, rtol, atol):
    """Taken from scipy select initial step part of the ode solver package. Slightly modified


    Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    interface : ModelInterface - contains the interface functions (internal/external) between model and solver

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

    interface.set_states(y0)
    f0 = interface.get_deriv(t0)

    if y0.size == 0:
        return np.inf

    scale = atol + np.abs(y0) * rtol
    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * direction * f0
    interface.set_states(y1)
    f1 = interface.get_deriv(t0 + h0 * direction)
    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1 / (order + 1))

    # Restore initial states
    interface.set_states(y0)

    return min(100 * h0, h1)


class Jacobian:
    """
    Jacobian base class
    """
    def get_jacobian(self, t):
        raise NotImplementedError


class UserDefinedJacobian(Jacobian):
    """
    Jacobian user defined class using ModelInterface get_jacobian(t)
    """
    def __init__(self, interface: ModelInterface):
        self.interface = interface

    def get_jacobian(self, t):
        return self.interface.get_jacobian(t)


class NumericalJacobian(Jacobian):
    """
    Numerical Jacobian class using functions to calculate the jacobian automatically.
    """

    def __init__(self, interface: ModelInterface, tolerance: float):
        self.interface = interface
        self.tolerance = tolerance
        self.factor = np.full(len(self.interface.get_states()), EPS ** 0.5)

    def _fun(self, t, y):
        """
        A vectorized derivative function
        :param t:
        :param y:
        :return:
        """
        f = np.empty_like(y)
        for i, yi in enumerate(y.T):
            self.interface.set_states(yi)
            f[:, i] = self.interface.get_deriv(t)
        return f

    def get_jacobian(self, t):
        """
        Wrapper that returns jacobian. Saves the factor internally.
        :param t:
        :return: jacobian matrix.
        """
        y = np.ascontiguousarray(self.interface.get_states())
        f = np.ascontiguousarray(self.interface.get_deriv(t))

        h, y_scale = self.get_jacobian_step_size(self.tolerance, y, f, self.factor)

        j, factor = self._num_jac(t, y, f, h, self.factor, y_scale)
        self.factor = factor

        return j

    def _num_jac(self, t, y, f, h, factor, y_scale):
        """
        The numerical jacobian method. Copied from SCIPY and modified to work with Numba and the ModelInterface
        :param t: time
        :param y: states
        :param f: derivatives
        :param h: step-size
        :param factor: a vector with length - updated on exit
        :param y_scale: scale of y
        :return: numerical jacobian and factor to update
        """

        n = y.shape[0]
        h_vecs = np.diag(h)

        f_new = self._fun(t, y.reshape((n, 1)) + h_vecs)
        diff = f_new - f.reshape((n, 1))
        max_ind = np.argmax(np.abs(diff), axis=0).astype('int')
        max_diff = np.diag(diff[max_ind]).copy()

        scale = np.maximum(np.abs(f[max_ind]), np.abs(np.diag(f_new[max_ind])))
        diff_too_small = max_diff < NUM_JAC_DIFF_REJECT * scale
        if np.any(diff_too_small):
            ind = np.nonzero(diff_too_small)[0].astype('int')
            new_factor = NUM_JAC_FACTOR_INCREASE * factor[ind]
            h_new = (y[ind] + new_factor * y_scale[ind]) - y[ind]
            for i, ind_ in enumerate(ind):
                h_vecs[ind_, ind_] = h_new[i]
            f_new = self._fun(t, y.reshape((n, 1)) + h_vecs[:, ind])
            diff_new = f_new - f.reshape((n, 1))
            max_ind = np.argmax(np.abs(diff_new), axis=0).astype('int')
            max_diff_new = np.diag(diff_new[max_ind])
            scale_new = np.maximum(np.abs(f[max_ind]), np.abs(np.diag(f_new[max_ind])))

            update = max_diff[ind] * scale_new < max_diff_new * scale[ind]
            if np.any(update):
                update = np.nonzero(update)[0].astype('int')
                update_ind = ind[update]
                factor[update_ind] = new_factor[update]
                h[update_ind] = h_new[update]
                diff[:, update_ind] = diff_new[:, update]
                scale[update_ind] = scale_new[update]
                max_diff[update_ind] = max_diff_new[update]

        diff /= h

        factor[max_diff < NUM_JAC_DIFF_SMALL * scale] *= NUM_JAC_FACTOR_INCREASE
        factor[max_diff > NUM_JAC_DIFF_BIG * scale] *= NUM_JAC_FACTOR_DECREASE
        factor = np.maximum(factor, NUM_JAC_MIN_FACTOR)

        return diff, factor

    @staticmethod
    def get_jacobian_step_size(threshold: float, y: np.array, f: np.array, factor: np.array):
        """
        helper function to determine the best step size of the jacobian. Step extracted from scipy num_jac.
        :param threshold:
        :param y:
        :param f:
        :param factor:
        :return:
        """

        f_sign = 2 * (np.real(f) >= 0).astype('float') - 1
        y_scale = f_sign * np.maximum(threshold, np.abs(y))
        h = (y + factor * y_scale) - y

        # Make sure that the step is not 0 to start with. Not likely it will be
        # executed often.
        for i in np.nonzero(h == 0)[0]:
            while h[i] == 0:
                factor[i] *= 10
                h[i] = (y[i] + factor[i] * y_scale[i]) - y[i]

        return h, y_scale


