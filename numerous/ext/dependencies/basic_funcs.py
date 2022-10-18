from numerous.multiphysics.equation_decorators import NumerousFunction
import numpy as np

J_MWh = 2.7777778e-10
Cp_W =4187
F_dT_3 = 1 / 12600
F_dT_5 = 1 / 20950
T_az = 273.15

@NumerousFunction()
def no_zero_div(a, b, d=0):

    if b == 0.0:
        return d
    else:
        return a / b

@NumerousFunction()
def h_T_water(T):
    return (T+T_az)*Cp_W


def add_power_with_energy(self, P_tag, init_val=0, scale=J_MWh):
    E_tag = 'E'+P_tag[1:]
    self.add_parameter(P_tag, init_val=init_val, integrate={'tag': E_tag, 'scale': scale})

@NumerousFunction()
def iterate_internal_temp(T_in, T_int, P, R, F):
    T_out = T_in + no_zero_div(P / Cp_W, F, 0)
    T_exp = P * R + T_out
    T_int_dot = (T_exp - T_int) * 0.1
    return T_int_dot, T_out, T_exp

@NumerousFunction()
def coerce(x, x1, x2):
    return max(min(x, max(x1,x2)), min(x1,x2))

@NumerousFunction()
def min_abs(x1, x2):
    return x1 if abs(x1)<abs(x2) else x2

@NumerousFunction()
def print_var(var):
    print(var)
    return var

@NumerousFunction()
def nan_zero(val):
    return 0 if np.isnan(val) else val

@NumerousFunction()
def dTLM(T_in_pri, T_out_pri, T_in_sec, T_out_sec):

    dTLM_ = nan_zero(no_zero_div(((T_in_pri - T_out_sec) - (T_out_pri - T_in_sec)), np.log(
        abs(no_zero_div(T_in_pri - T_out_sec, T_out_pri - T_in_sec, 0))),0))

    return (T_in_pri - T_out_sec + T_out_pri - T_in_sec)/2 if dTLM_ == 0 else dTLM_


