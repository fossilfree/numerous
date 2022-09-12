import ctypes

import numpy as np
from numba import njit


def get_fmu_functions(fmu, fmu_logging=False):
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
    if fmu_logging:
        @njit()
        def wr_set_time(_component, t):
            output = "[FMI] fmi2SetTime(component=" + str(_component) + ", time=" + float_to_str(t) + ")"
            print(output)
            set_time(_component, t)

        @njit()
        def wr_completedIntegratorStep(_component, id, event, term):
            output = "[FMI] fmi2CompletedIntegratorStep(component=" + str(_component) + "," \
                                                                                        "noSetFMUStatePriorToCurrentPoint=" \
                     + str(id) + ", enterEventMode=" + \
                     str(0) + ", terminateSimulation=" + str(0) + ")"
            print(output)
            completedIntegratorStep(_component, id, event, term)

        @njit()
        def wr_enter_cont_mode(_component):
            q = "[FMI] fmi2EnterContinuousTimeMode(component=" + str(_component) + ")"
            print(q)
            enter_cont_mode(_component)

        @njit()
        def wr_getreal(_component, vr, len_q, value):
            print("[FMI] fmi2GetReal(component=" + str(_component) + ", vr=")
            print(vr)
            print(", nvr=" + str(len_q) + ",value=")
            print(value)
            print(")")
            getreal(_component, vr.ctypes, len_q, value.ctypes)

        @njit()
        def wr_fmi2SetReal(_component, vr, len_q, value1):
            print("[FMI] fmi2SetReal(component=" + str(_component) + ", vr=")
            print(vr)
            print(", nvr=" + str(len_q) + ",value=")
            print(value1)
            print(")")
            fmi2SetReal(_component, vr.ctypes, len_q, value1.ctypes)

        @njit()
        def wr_fmi2SetC(_component, value3, i):
            print("[FMI] fmi2SetContinuous(component=" + str(_component) + ", x=")
            print(value3)
            print(", nx=" + str(i) + ")")
            fmi2SetC(_component, value3.ctypes, i)

        @njit()
        def wr_get_event_indicators(_component, value_event, event_n):
            print("[FMI] fmi2GetEventIndicators(component=" + str(_component) + ", eventIndicators=")
            print(value_event)
            print(", ni=" + str(event_n) + ")")

            return get_event_indicators(_component, value_event.ctypes, event_n)

        @njit()
        def wr_enter_event_mode(_component):
            q = "[FMI] fmi2EnterEventMode(component=" + str(_component) + ")"
            print(q)
            enter_event_mode(_component)

        @njit()
        def wr_newDiscreteStates(_component, q_a):
            q = "[FMI] NewDiscreteStates(component=" + str(_component) + ",q_a_ptr)"
            print(q)
            newDiscreteStates(_component, q_a)

        return wr_getreal, wr_set_time, wr_fmi2SetC, wr_fmi2SetReal, wr_completedIntegratorStep, \
               wr_get_event_indicators, wr_enter_event_mode, wr_enter_cont_mode, wr_newDiscreteStates
    else:
        @njit()
        def wr_getreal(_component, vr, len_q, value):
            getreal(_component, vr.ctypes, len_q, value.ctypes)

        @njit()
        def wr_fmi2SetReal(component, vr, len_q, value1):
            fmi2SetReal(component, vr.ctypes, len_q, value1.ctypes)

        @njit()
        def wr_fmi2SetC(component, value3, i):
            fmi2SetC(component, value3.ctypes, i)

        @njit()
        def wr_get_event_indicators(component, value_event, event_n):
            return get_event_indicators(component, value_event.ctypes, event_n)

        return wr_getreal, set_time, wr_fmi2SetC, wr_fmi2SetReal, completedIntegratorStep, \
               wr_get_event_indicators, enter_event_mode, enter_cont_mode, newDiscreteStates
