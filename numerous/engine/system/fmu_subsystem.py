import ctypes

from numerous import EquationBase
from numerous.engine.system import Subsystem
from numba import cfunc, carray
import numpy as np


class FMU_Subsystem(Subsystem, EquationBase):
    """
    """

    def __init__(self, filename: str, tag: str):
        super().__init__(tag)
        self.add_parameter('e', 0.7)
        self.add_constant('g', 9.81)
        self.add_state('h', 1)
        self.add_state('v', 0)

        getreal = getattr(fmu.dll, "fmi2GetReal")
        component = fmu.component
        vr = np.array([1], dtype=np.int32)
        sig = "uint32(uint32,CPointer(uint32),uint32,CPointer(float64))"

        value = np.empty(1).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        getreal.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
        getreal.restype = ctypes.c_uint

        set_time = getattr(fmu.dll, "fmi2SetTime")
        set_time.argtypes = [ctypes.c_void_p, ctypes.c_double]
        set_time.restype = ctypes.c_int

        fmi2SetReal = getattr(fmu.dll, "fmi2SetReal")
        fmi2SetReal.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
        fmi2SetReal.restype = ctypes.c_uint

        completedIntegratorStep = getattr(fmu.dll, "fmi2CompletedIntegratorStep")
        completedIntegratorStep.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        completedIntegratorStep.restype = ctypes.c_uint

        eval_llvm_signature = 'void(CPointer(int32),CPointer(int32),CPointer(float64),CPointer(float64),CPointer(float64),CPointer(float64),CPointer(' \
                              'float64),CPointer(float64),float64,float64,float64,float64,float64,float64,float64) '
        len_q = 6

        term_1 = ctypes.byref(ctypes.c_int(0))
        event_1 = ctypes.byref(ctypes.c_int(0))

        def eval_llvm(event, term, a0, a1, a2, a3, a4, a5, a_i_0, a_i_1, a_i_2, a_i_3, a_i_4, a_i_5, t):
            vr = np.arange(0, len_q, 1, dtype=np.uint32)
            value = np.zeros(len_q, dtype=np.float64)
            value1 = np.array([a_i_0, a_i_1, a_i_2, a_i_3, a_i_4, a_i_5], dtype=np.float64)
            fmi2SetReal(component, vr.ctypes, len_q, value1.ctypes)
            set_time(component, t)
            completedIntegratorStep(component, 1, event, term)
            getreal(component, vr.ctypes, len_q, value.ctypes)
            carray(a0, (1,))[0] = value[0]
            carray(a1, (1,))[0] = value[1]
            carray(a2, (1,))[0] = value[2]
            carray(a3, (1,))[0] = value[3]
            carray(a4, (1,))[0] = value[4]
            carray(a5, (1,))[0] = value[5]

        fmu.enterContinuousTimeMode()
        equation_call = cfunc(sig=eval_llvm_signature)(eval_llvm)


fmu_filename = 'bouncingBall.fmu'
fmu_subsystem = FMU_Subsystem(fmu_filename, "BouncingBall")
