import ctypes

import numpy as np
from numba import njit


def get_fmu_functions(fmu, logging=True):
    @njit()
    def float_to_str(n):
        int_part = int(n)
        float_part = n - int(n)
        float_part = float_part * pow(10, 5)
        float_part = int(float_part)
        return str(int_part) + "." + str(int(float_part))

    getreal = getattr(fmu.dll, "fmi2GetReal")
    getreal.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    getreal.restype = ctypes.c_uint

    set_time = getattr(fmu.dll, "fmi2SetTime")
    set_time.argtypes = [ctypes.c_void_p, ctypes.c_double]
    set_time.restype = ctypes.c_int

    fmi2SetC = getattr(fmu.dll, "fmi2SetContinuousStates")
    fmi2SetC.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fmi2SetC.restype = ctypes.c_uint

    fmi2SetReal = getattr(fmu.dll, "fmi2SetReal")
    fmi2SetReal.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    fmi2SetReal.restype = ctypes.c_uint

    completedIntegratorStep = getattr(fmu.dll, "fmi2CompletedIntegratorStep")
    completedIntegratorStep.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_void_p]
    completedIntegratorStep.restype = ctypes.c_uint

    get_event_indicators = getattr(fmu.dll, "fmi2GetEventIndicators")

    get_event_indicators.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    get_event_indicators.restype = ctypes.c_uint

    enter_event_mode = getattr(fmu.dll, "fmi2EnterEventMode")

    enter_event_mode.argtypes = [ctypes.c_void_p]
    enter_event_mode.restype = ctypes.c_uint

    enter_cont_mode = getattr(fmu.dll, "fmi2EnterContinuousTimeMode")
    enter_cont_mode.argtypes = [ctypes.c_void_p]
    enter_cont_mode.restype = ctypes.c_uint

    newDiscreteStates = getattr(fmu.dll, "fmi2NewDiscreteStates")
    newDiscreteStates.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    newDiscreteStates.restype = ctypes.c_uint
    if logging:
        @njit()
        def wr_set_time(_component, t):
            q = "[FMI] fmi2SetTime(component=" + float_to_str(_component) + ", time=" + float_to_str(t) + ")"
            print(q)
            set_time(_component, t)

        @njit()
        def wr_enter_cont_mode(_component):
                q = "[FMI] fmi2EnterContinuousTimeMode(component=" + float_to_str(_component) + ")"
                print(q)
                enter_cont_mode(_component)

        @njit()
        def wr_getreal(_component, vr, len_q, value):
            # j  = ctypes.cast(vr.data, ctypes.POINTER(ctypes.c_double*len_q))[0]
            # print(np.frombuffer(j[0], np.float64)[0])
            q = "[FMI] fmi2GetReal"
            print(vr)
            getreal(_component, vr.ctypes, len_q, value)

        return wr_getreal, wr_set_time, fmi2SetC, fmi2SetReal, completedIntegratorStep, \
               get_event_indicators, enter_event_mode, wr_enter_cont_mode, newDiscreteStates
    else:
        return getreal, set_time, fmi2SetC, fmi2SetReal, completedIntegratorStep, \
               get_event_indicators, enter_event_mode, enter_cont_mode, newDiscreteStates
