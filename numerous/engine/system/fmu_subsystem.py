import ctypes

from fmpy import read_model_description, extract

from fmpy.fmi1 import printLogMessage
from fmpy.fmi2 import FMU2Model, fmi2CallbackFunctions, fmi2CallbackLoggerTYPE, fmi2CallbackAllocateMemoryTYPE, \
    fmi2CallbackFreeMemoryTYPE, allocateMemory, freeMemory
from fmpy.simulation import apply_start_values, Input
from fmpy.util import auto_interval

from numerous import EquationBase, Equation
from numerous.engine.model import Model
from numerous.engine.system import Subsystem, Item
from numba import cfunc, carray
import numpy as np

class FMU_Subsystem(Subsystem, EquationBase):
    """
    """

    def __init__(self, fmu_filename: str, tag: str):
        super().__init__(tag)
        self.add_parameter('e', 0.7)
        self.add_constant('g', 9.81)
        self.add_state('h', 1)
        self.add_state('v', 0)
        input = None
        fmi_call_logger = None
        start_values = {}
        validate = False
        step_size = None
        relative_tolerance = None
        output_interval = None
        start_values = start_values
        apply_default_start_values = False
        input = input
        debug_logging = False
        visible = False
        model_description = read_model_description(fmu_filename, validate=validate)
        required_paths = ['resources', 'binaries/']
        tempdir = extract(fmu_filename, include=lambda n: n.startswith(tuple(required_paths)))
        unzipdir = tempdir
        fmi_type = "ModelExchange"
        experiment = model_description.defaultExperiment
        start_time = 0.0
        start_time = float(start_time)

        stop_time = start_time + 1.0

        stop_time = float(stop_time)

        if relative_tolerance is None and experiment is not None:
            relative_tolerance = experiment.tolerance

        if step_size is None:
            total_time = stop_time - start_time
            step_size = 10 ** (np.round(np.log10(total_time)) - 3)

        if output_interval is None and fmi_type == 'CoSimulation' and experiment is not None and experiment.stepSize is not None:
            output_interval = experiment.stepSize
            while (stop_time - start_time) / output_interval > 1000:
                output_interval *= 2

        fmu_args = {
            'guid': model_description.guid,
            'unzipDirectory': unzipdir,
            'instanceName': None,
            'fmiCallLogger': fmi_call_logger
        }
        logger = printLogMessage
        callbacks = fmi2CallbackFunctions()
        callbacks.logger = fmi2CallbackLoggerTYPE(logger)
        callbacks.allocateMemory = fmi2CallbackAllocateMemoryTYPE(allocateMemory)
        callbacks.freeMemory = fmi2CallbackFreeMemoryTYPE(freeMemory)

        fmu_args['modelIdentifier'] = model_description.modelExchange.modelIdentifier

        fmu = FMU2Model(**fmu_args)
        fmu.instantiate(visible=visible, callbacks=callbacks, loggingOn=debug_logging)

        if relative_tolerance is None:
            relative_tolerance = 1e-5
        if output_interval is None:
            if step_size is None:
                output_interval = auto_interval(stop_time - start_time)
            else:
                output_interval = step_size
                while (stop_time - start_time) / output_interval > 1000:
                    output_interval *= 2

        if step_size is None:
            step_size = output_interval
            max_step = (stop_time - start_time) / 1000
            while step_size > max_step:
                step_size /= 2

        ##this call should happen at init at model.py
        fmu.setupExperiment(startTime=start_time, stopTime=stop_time)

        ##

        input = Input(fmu, model_description, input)

        apply_start_values(fmu, model_description, start_values, apply_default_start_values)

        fmu.enterInitializationMode()
        input.apply(start_time)
        fmu.exitInitializationMode()

        getreal = getattr(fmu.dll, "fmi2GetReal")
        component = fmu.component

        getreal.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
        getreal.restype = ctypes.c_uint

        set_time = getattr(fmu.dll, "fmi2SetTime")
        set_time.argtypes = [ctypes.c_void_p, ctypes.c_double]
        set_time.restype = ctypes.c_int

        fmi2SetReal = getattr(fmu.dll, "fmi2SetReal")
        fmi2SetReal.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p]
        fmi2SetReal.restype = ctypes.c_uint

        completedIntegratorStep = getattr(fmu.dll, "fmi2CompletedIntegratorStep")
        completedIntegratorStep.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int),
                                            ctypes.POINTER(ctypes.c_int)]
        completedIntegratorStep.restype = ctypes.c_uint

        eval_llvm_signature = 'list(CPointer(float64))(CPointer(int32),CPointer(int32),CPointer(float64),CPointer(float64),CPointer(float64),CPointer(float64),CPointer(' \
                              'float64),CPointer(float64),float64,float64,float64,float64,float64,float64,float64) '
        len_q = 6

        self.term_1 = ctypes.byref(ctypes.c_int(0))
        self.event_1 = ctypes.byref(ctypes.c_int(0))

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
            return  np.array([a1,a3], dtype=np.float64)

        fmu.enterContinuousTimeMode()

        self.equation_call = cfunc(sig=eval_llvm_signature)(eval_llvm)

        self.t1 = self.create_namespace('t1')
        self.t1.add_equations([self])

    @Equation()
    def eval(self, scope):
        a0 = np.array([scope.e])
        a4 = np.array([scope.g])
        a3 = np.array([scope.h])
        a2 = np.array([scope.v])
        a1 = np.array([0.0])
        a5 = np.array([0.0])
        a1,a3=self.equation_call(self.event_1, self.term_1, a0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           a1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           a2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           a3.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           a4.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                           a5.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), scope.h, 0.0, scope.v, 0.0, scope.g, scope.e,
                           0.1)
        scope.h_dot = a1
        scope.v_dot = a3

class Test_Eq(EquationBase):
    __test__ = False

    def __init__(self, T=0, R=1):
        super().__init__(tag='T_eq')
        self.add_state('Q', T)
        self.add_parameter('R', R)

    @Equation()
    def eval(self, scope):
        scope.Q_dot = scope.R + 9


class G(Item):
    def __init__(self, tag, TG, RG):
        super().__init__(tag)
        t1 = self.create_namespace('t1')
        t1.add_equations([Test_Eq(T=TG, R=RG)])


class S3(Subsystem):
    def __init__(self, tag):
        super().__init__(tag)

        fmu_filename = 'bouncingBall.fmu'
        fmu_subsystem = FMU_Subsystem(fmu_filename, "BouncingBall")
        item_t = G('test', TG=10, RG=2)
        item_t.t1.R = fmu_subsystem.t1.h
        self.register_items([fmu_subsystem, item_t])


subsystem1 = S3('q1')
m1 = Model(subsystem1, use_llvm=False)
print("finished")
