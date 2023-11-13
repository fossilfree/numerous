from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any
from uuid import uuid4

from numerous.engine import VariableType
from .signal import Signal, default_signal


@dataclasses.dataclass
class VariableAttrs:
    name: str = None
    signal: Signal = default_signal
    _host: Any = None
    _host_attr: str = None
    id: str = dataclasses.field(default_factory=lambda: str(uuid4()))
    value: float = 0.0
    is_deriv: bool = False
    is_instance: bool = False
    cls: object = None
    integrate: dict = None
    type: VariableType = VariableType.PARAMETER
    var_instance: Variable = None
    construct: bool = True
    fixed: bool = False
    additive: bool = True
    must_map: bool = False
    mapped_to: list = dataclasses.field(default_factory=lambda: list())


variable_attrs_fields = [f.name for f in dataclasses.fields(VariableAttrs)]


class PartialResult:

    def __init__(self, *args, op=None, func=None):
        self.id = str(uuid4())

        self.arguments = args

        self.op = op
        self.func = func

    def __add__(self, other):
        return PartialResult(self, other, op=Operations.ADD)

    def __sub__(self, other):
        return PartialResult(self, other, op=Operations.SUB)

    def __mul__(self, other):
        return PartialResult(self, other, op=Operations.MUL)

    def __truediv__(self, other):
        return PartialResult(self, other, op=Operations.DIV)

    def __neg__(self):
        return PartialResult(self, op=Operations.NEG)

    def __lt__(self, other):
        return PartialResult(self, other, op=Operations.LT)

    def __gt__(self, other):
        return PartialResult(self, other, op=Operations.GT)

    def __eq__(self, other):
        return PartialResult(self, other, op=Operations.EQ)

    def clone(self, variables):

        args_ = []
        for arg in self.arguments:
            if isinstance(arg, Variable):

                a = getattr(variables, arg.name)
            else:
                a = arg._clone(variables)

            args_.append(a)

        return PartialResult(*args_, op=self.op, func=self.func)


class Variable(PartialResult):
    """
    Declation of a variable
    """

    def __init__(self, **kwargs):

        kwargs_ = {}
        self._variable = None

        for k, v in kwargs.items():
            if v is not None:
                kwargs_[k] = v

        self._attrs = VariableAttrs(**kwargs_)

    def __setattr__(self, key, value):
        if key in variable_attrs_fields:
            self._attrs.__setattr__(key, value)
        else:
            super(Variable, self).__setattr__(key, value)

    def __getattr__(self, item):
        if item in variable_attrs_fields:
            return self._attrs.__getattribute__(item)
        else:
            return super(Variable, self).__getattribute__(item)

    def set_variable(self, var):
        if self._variable:
            raise ValueError(f'Variable already set! {self._variable}')
        self._variable = var

    def clone(self, id, name=None, is_instance=False, host=None):

        return Variable(id=id, value=self.value, name=self.name if name is None else name, is_deriv=self.is_deriv,
                        is_instance=is_instance, type=self.type,
                        var_instance=self.var_instance, integrate=self.integrate, construct=self.construct,
                        fixed=self.fixed, mapped_to=self.mapped_to, must_map=self.must_map,
                        _host=host if host is not None else self._host
                        )

    def instance(self, id, name, host):
        if self.is_instance:
            raise ValueError('Cannot instance from an instance')
        instance = self.clone(id=id, name=name, is_instance=True, host=host)
        instance.cls = self
        return instance

    def __repr__(self):
        return f"{self.name}, {self.id}"

    def __eq__(self, other: Variable):
        if hasattr(other, 'id'):
            return self.id == other.id
        else:
            return False

    def get_path(self, parent):

        path = self._host.get_path(parent)

        return path + [self._host_attr]

    def set_host(self, host, attr):
        self._host = host
        self._host_attr = attr


class Parameter(Variable):
    """
    Declaration of a Parameter
    """

    def __init__(self, value, id=None, name=None, integrate=None, must_be_mapped=False, is_fixed=False):
        super(Parameter, self).__init__(value=value, id=id, name=name, is_deriv=False, type=VariableType.PARAMETER,
                                        integrate=integrate, must_map=must_be_mapped, fixed=is_fixed)


class Constant(Variable):
    """
    Declaration of a Constant. A constant cannot be changed.
    """

    def __init__(self, value, id=None, name=None):
        super(Constant, self).__init__(value=value, id=id, name=name, is_deriv=False, type=VariableType.CONSTANT,
                                       fixed=True)


def State(value):
    """
    Declaration of a State. States have time derivatives which are integrated to the state values of the system.
    """
    return Variable(value=value, type=VariableType.STATE), Variable(value=0, type=VariableType.DERIVATIVE)


def integrate(var, integration_spec):
    var.integrate = integration_spec
    return var, Variable(value=0, construct=False)


class Operations(Enum):
    ADD = 1
    SUB = 2
    DIV = 3
    MUL = 4
    POW = 5
    FUNC = 6
    NEG = 7
    LT = 8
    GT = 9
    GET = 10
    LET = 11
    EQ = 12
