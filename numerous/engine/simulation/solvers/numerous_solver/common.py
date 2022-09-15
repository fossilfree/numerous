import numpy as np
from .interface import ModelInterface

def norm(x):
    return np.linalg.norm(x) / len(x) ** 0.5

def select_initial_step(interface: ModelInterface, t0, y0, direction, order, rtol, atol):
    """Taken from scipy select initial step part of the ode solver package. Slightly modified


    Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    interface : NumerousSolverInterface - contains the interface functions (internal/external) between model and solver

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
