import ctypes
import os
from typing import List

from attr import attrs, attrib, Factory
from fmpy import read_model_description, extract

from fmpy.fmi1 import printLogMessage
from fmpy.fmi2 import FMU2Model, fmi2CallbackFunctions, fmi2CallbackLoggerTYPE, fmi2CallbackAllocateMemoryTYPE, \
    fmi2CallbackFreeMemoryTYPE, allocateMemory, freeMemory, fmi2EventInfo
from fmpy.simulation import apply_start_values, Input
from fmpy.util import auto_interval
from numba.core.extending import intrinsic

from numerous import EquationBase, Equation, NumerousFunction
from numerous.engine.model import Model
from numerous.engine.simulation import Simulation, SolverType
from numerous.engine.system import Subsystem, Item
from numba import cfunc, carray, types, njit
import numpy as np


@attrs(eq=False)
class ScalarVariable:
    name = attrib(type=str)
    variability = attrib(type=str, default=None, repr=False)
    initial = attrib(type=str, default=None, repr=False)


@attrs(eq=False)
class ModelDescription(object):
    modelVariables = attrib(type=List[ScalarVariable], default=Factory(list), repr=False)


class FMU_Subsystem(Subsystem, EquationBase):
    """
    """

    def __init__(self, fmu_filename: str, tag: str, h: float):
        super().__init__(tag)
        # self.add_parameter('e', 0.7)
        # self.add_constant('g', 9.81)
        # self.add_state('h', h)
        # self.add_state('v', 0)
        self.model_description = None
        input = None
        fmi_call_logger = None
        start_values = {}
        validate = False
        step_size = None
        output_interval = None
        start_values = start_values
        apply_default_start_values = False
        input = input
        debug_logging = False
        visible = False
        model_description = read_model_description(fmu_filename, validate=validate)
        self.model_description = self.read_model_description(fmu_filename)
        self.set_variables(self.model_description)
        required_paths = ['resources', 'binaries/']
        tempdir = extract(fmu_filename, include=lambda n: n.startswith(tuple(required_paths)))
        unzipdir = tempdir
        fmi_type = "ModelExchange"
        experiment = model_description.defaultExperiment
        start_time = 0.0
        start_time = float(start_time)

        stop_time = start_time + 1.0

        stop_time = float(stop_time)

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
        completedIntegratorStep.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                            ctypes.c_void_p]
        completedIntegratorStep.restype = ctypes.c_uint

        len_q = 6

        def eval_llvm(event, term, a0, a1, a2, a3, a4, a5, a_i_0, a_i_2, a_i_4, a_i_5, t):
            vr = np.arange(0, len_q, 1, dtype=np.uint32)
            value = np.zeros(len_q, dtype=np.float64)
            ## we are reading derivatives from FMI
            getreal(component, vr.ctypes, len_q, value.ctypes)
            value1 = np.array([a_i_0, value[1], a_i_2, value[3], a_i_4, a_i_5], dtype=np.float64)
            fmi2SetReal(component, vr.ctypes, len_q, value1.ctypes)
            set_time(component, t)
            completedIntegratorStep(component, 1, event, term)
            getreal(component, vr.ctypes, len_q, value.ctypes)
            carray(a0, (1,), dtype=np.float64)[0] = value[0]
            carray(a1, (1,), dtype=np.float64)[0] = value[1]
            carray(a2, (1,), dtype=np.float64)[0] = value[2]
            carray(a3, (1,), dtype=np.float64)[0] = value[3]
            carray(a4, (1,), dtype=np.float64)[0] = value[4]
            carray(a5, (1,), dtype=np.float64)[0] = value[5]

        fmu.enterContinuousTimeMode()

        self.t1 = self.create_namespace('t1')
        self.t1.add_equations([self])
        equation_call = cfunc(types.void(types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.voidptr
                                         , types.voidptr, types.voidptr, types.voidptr, types.float64, types.float64,
                                         types.float64, types.float64, types.float64))(eval_llvm)

        term_1 = np.array([0], dtype=np.int32)
        term_1_ptr = term_1.ctypes.data
        event_1 = np.array([0], dtype=np.int32)
        event_1_ptr = event_1.ctypes.data

        a0 = np.array([0], dtype=np.float64)
        a0_ptr = a0.ctypes.data

        a1 = np.array([0], dtype=np.float64)
        a1_ptr = a1.ctypes.data

        a2 = np.array([0], dtype=np.float64)
        a2_ptr = a2.ctypes.data

        a3 = np.array([0], dtype=np.float64)
        a3_ptr = a3.ctypes.data

        a4 = np.array([0], dtype=np.float64)
        a4_ptr = a4.ctypes.data

        a5 = np.array([0], dtype=np.float64)
        a5_ptr = a5.ctypes.data

        @intrinsic
        def address_as_void_pointer(typingctx, src):
            """ returns a void pointer from a given memory address """
            from numba.core import types, cgutils
            sig = types.voidptr(src)

            def codegen(cgctx, builder, sig, args):
                return builder.inttoptr(args[0], cgutils.voidptr_t)

            return sig, codegen

        @NumerousFunction()
        def fmu_eval(e, g, h, v):
            carray(address_as_void_pointer(a0_ptr), a0.shape, dtype=a0.dtype)[0] = 0
            carray(address_as_void_pointer(a4_ptr), a4.shape, dtype=a0.dtype)[0] = 0
            carray(address_as_void_pointer(a3_ptr), a3.shape, dtype=a0.dtype)[0] = 0
            carray(address_as_void_pointer(a2_ptr), a3.shape, dtype=a0.dtype)[0] = 0
            carray(address_as_void_pointer(a1_ptr), a3.shape, dtype=a0.dtype)[0] = 0
            carray(address_as_void_pointer(a5_ptr), a3.shape, dtype=a0.dtype)[0] = 0
            equation_call(address_as_void_pointer(event_1_ptr),
                          address_as_void_pointer(term_1_ptr),
                          address_as_void_pointer(a0_ptr),
                          address_as_void_pointer(a1_ptr),
                          address_as_void_pointer(a2_ptr),
                          address_as_void_pointer(a3_ptr),
                          address_as_void_pointer(a4_ptr),
                          address_as_void_pointer(a5_ptr), h, v, g, e,
                          0.1)

            return carray(address_as_void_pointer(a1_ptr), (1,), dtype=np.float64)[0], \
                   carray(address_as_void_pointer(a3_ptr), (1,), dtype=np.float64)[0]

        self.fmu_eval = fmu_eval

        get_event_indicators = getattr(fmu.dll, "fmi2GetEventIndicators")

        get_event_indicators.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        get_event_indicators.restype = ctypes.c_uint
        event_n = 1

        # returns position to find zero crossing using root finding algorithm of scipy solver
        def hitground_event_fun(event_indicators, t, y1, y2):

            value_event = np.zeros(event_n, dtype=np.float64)

            vr = np.arange(0, len_q, 1, dtype=np.uint32)
            value = np.zeros(len_q, dtype=np.float64)
            getreal(component, vr.ctypes, len_q, value.ctypes)
            value1 = np.array([y1, value[1], y2, value[3], value[4], value[5]], dtype=np.float64)
            fmi2SetReal(component, vr.ctypes, len_q, value1.ctypes)
            set_time(component, t)
            get_event_indicators(component, value_event.ctypes, event_n)
            value2 = np.array([value[0], value[1], value[2], value[3], value[4], value[5]], dtype=np.float64)
            fmi2SetReal(component, vr.ctypes, len_q, value2.ctypes)

            carray(event_indicators, (1,), dtype=np.float64)[0] = value_event[0]

        event_ind_call_1 = cfunc(types.void(types.voidptr, types.float64, types.float64, types.float64))(
            hitground_event_fun)

        c = np.array([0], dtype=np.float64)
        c_ptr = a3.ctypes.data

        @njit
        def event_cond(t, y):
            temp_addr = address_as_void_pointer(c_ptr)
            carray(temp_addr, a0.shape, dtype=a0.dtype)[0] = 0
            event_ind_call_1(temp_addr, t, y[0], y[1])
            result = carray(temp_addr, (1,), dtype=np.float64)[0]
            return result

        def event_cond_2(t, variables):
            q = np.array([variables['t1.h'], variables['t1.v']])
            return event_cond(t, q)

        enter_event_mode = getattr(fmu.dll, "fmi2EnterEventMode")

        enter_event_mode.argtypes = [ctypes.c_uint]
        enter_event_mode.restype = ctypes.c_uint

        enter_cont_mode = getattr(fmu.dll, "fmi2EnterContinuousTimeMode")
        enter_cont_mode.argtypes = [ctypes.c_uint]
        enter_cont_mode.restype = ctypes.c_uint

        newDiscreteStates = getattr(fmu.dll, "fmi2NewDiscreteStates")
        newDiscreteStates.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        newDiscreteStates.restype = ctypes.c_uint

        # change direction of movement upon event detection and reduce velocity
        def hitground_event_callback_fun(q, a_e):
            enter_event_mode(component)
            newDiscreteStates(component, q)
            enter_cont_mode(component)
            vr = np.arange(0, len_q, 1, dtype=np.uint32)
            value = np.zeros(len_q, dtype=np.float64)
            getreal(component, vr.ctypes, len_q, value.ctypes)
            carray(a_e, (1,), dtype=np.float64)[0] = value[2]

        eventInfo = fmi2EventInfo()
        event_ind_call = cfunc(types.void(types.voidptr, types.voidptr))(hitground_event_callback_fun)
        q_ptr = ctypes.addressof(eventInfo)

        a_e = np.array([0], dtype=np.float64)
        a_e_ptr = a_e.ctypes.data

        @njit
        def event_action(x, y):
            carray(address_as_void_pointer(a_e_ptr), a_e.shape, dtype=a0.dtype)[0] = 0
            event_ind_call(address_as_void_pointer(q_ptr), address_as_void_pointer(a_e_ptr))
            return carray(address_as_void_pointer(a_e_ptr), (1,), dtype=np.float64)[0]

        def event_action_2(t, variables):
            q = np.array([variables['t1.h'], variables['t1.v']])
            velocity = event_action(t, q)
            variables['t1.v'] = velocity

        self.add_event("hitground_event", event_cond_2, event_action_2, compiled_functions={"event_cond": event_cond,
                                                                                            "event_action": event_action})

    @Equation()
    def eval(self, scope):
        scope.h_dot, scope.v_dot = self.fmu_eval(scope.e, scope.g, scope.h, scope.v)

    def set_variables(self, model_description):
        for variable in model_description.modelVariables:
            if variable.initial == 'exact':
                if variable.variability == 'fixed':
                    self.add_constant(variable.name, float(variable.start))
                if variable.variability == 'continuous':
                    self.add_state(variable.name, float(variable.start))
                if variable.variability == 'tunable':
                    self.add_parameter(variable.name, float(variable.start))

    @staticmethod
    def read_model_description(filename):
        import zipfile
        from lxml import etree

        # remember the original filename
        _filename = filename
        if isinstance(filename, str) and os.path.isdir(filename):  # extracted FMU
            filename = os.path.join(filename, 'modelDescription.xml')
            tree = etree.parse(filename)
        elif isinstance(filename, str) and os.path.isfile(filename) and filename.lower().endswith('.xml'):  # XML file
            tree = etree.parse(filename)
        else:  # FMU as path or file like object
            with zipfile.ZipFile(filename, 'r') as zf:
                xml = zf.open('modelDescription.xml')
                tree = etree.parse(xml)

        root = tree.getroot()

        fmiVersion = root.get('fmiVersion')

        is_fmi1 = fmiVersion == '1.0'
        is_fmi2 = fmiVersion == '2.0'
        is_fmi3 = fmiVersion.startswith('3.0')

        if not is_fmi1 and not is_fmi2 and not is_fmi3:
            raise Exception("Unsupported FMI version: %s" % fmiVersion)

        modelDescription = ModelDescription()
        for variable in root.find('ModelVariables'):
            if variable.get("name") is None:
                continue

            sv = ScalarVariable(name=variable.get('name'))
            sv.variability = variable.get('variability')
            sv.initial = variable.get('initial')

            if fmiVersion in ['1.0', '2.0']:
                # get the nested "value" element
                for child in variable.iterchildren():
                    if child.tag in {'Real', 'Integer', 'Boolean', 'String', 'Enumeration'}:
                        value = child
                        break
            else:
                value = variable

            if variable.tag == 'String':
                # handle <Start> element of String variables in FMI 3
                sv.start = variable.find('Start').get('value')
            else:
                sv.start = value.get('start')

            modelDescription.modelVariables.append(sv)
        return modelDescription


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
        fmu_subsystem = FMU_Subsystem(fmu_filename, "BouncingBall", h=1.6)
        fmu_subsystem2 = FMU_Subsystem(fmu_filename, "BouncingBall2", h=2)
        # fmu_subsystem3 = FMU_Subsystem(fmu_filename, "BouncingBall3", h=1.5)
        # item_t = G('test', TG=10, RG=2)
        # item_t.t1.R = fmu_subsystem.t1.h
        self.register_items([fmu_subsystem, fmu_subsystem2])


subsystem1 = S3('q1')
m1 = Model(subsystem1, use_llvm=True)
s = Simulation(
    m1, t_start=0, t_stop=1.0, num=500, num_inner=100, max_step=.1, solver_type=SolverType.NUMEROUS)
s.solve()
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
# t = np.linspace(0, 1.0, 100 + 1)
y = np.array(m1.historian_df["q1.BouncingBall.t1.h"])
y2 = np.array(m1.historian_df["q1.BouncingBall2.t1.h"])
# y3 = np.array(m1.historian_df["q1.BouncingBall3.t1.h"])
t = np.array(m1.historian_df["time"])
ax.plot(t, y)
ax.plot(t, y2)
# ax.plot(t, y3)

ax.set(xlabel='time (s)', ylabel='h',
       title='BB')
ax.grid()

plt.show()
