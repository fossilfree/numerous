import uuid
from enum import Enum
from functools import wraps
import inspect


class NumbaCallback(object):
    """

    """

    def __init__(self, method_type, run_after_init=False):
        """
        Parameters
        ----------
        method_type : CallbackMethodType
            type of method
        run_after_init : bool
            If True runs update right after initialisation of a callback before the first solver iteration.

        """
        self.method_type = method_type
        self.run_after_init = run_after_init
        self.id = str(uuid.uuid1())

    def __call__(self, func):
        @wraps(func)
        def wrapper(f_self, scope):
            func(f_self, scope)
        wrapper.run_after_init  = self.run_after_init
        wrapper.lines = inspect.getsource(func)
        wrapper.id = self.id
        return wrapper


class CallbackMethodType(Enum):
    INITIALIZE = 0
    UPDATE = 1
