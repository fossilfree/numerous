import types as ptypes
import ctypes
import ast

import pandas
from fmpy import read_model_description, extract

from fmpy.fmi1 import printLogMessage
from fmpy.fmi2 import FMU2Model, fmi2CallbackFunctions, fmi2CallbackLoggerTYPE, fmi2CallbackAllocateMemoryTYPE, \
    fmi2CallbackFreeMemoryTYPE, allocateMemory, freeMemory
from fmpy.simulation import apply_start_values, Input
from fmpy.util import auto_interval

from numerous.engine.system.external_mappings import InterpolationType, ExternalMappingElement
from numerous.engine.system.fmu_system_generator.fmu_functions_wrapper import get_fmu_functions
from numerous.multiphysics import EquationBase, Equation, NumerousFunction

from numerous.engine.system import Subsystem, ExternalMappingUnpacked
from numba import cfunc, carray, types, njit
import numpy as np

from numerous.engine.system.fmu_system_generator.fmu_ast_generator import generate_fmu_eval, generate_eval_llvm, \
    generate_eval_event, generate_njit_event_cond, generate_action_event, generate_event_action, generate_eq_call
from numerous.engine.system.fmu_system_generator.utils import address_as_void_pointer
from numerous.utils.data_loader import InMemoryDataLoader


def _replace_name_str(str_v):
    return str_v.replace(".", "_").replace("[", "_").replace("]", "_").replace(",", "_")


class FMU_Subsystem(Subsystem, EquationBase):
    """
    """

    def __init__(self, fmu_filename: str, tag: str, fmu_in: str = None, debug_output=False,
                 import_all=False, fmu_logging=False):
        super().__init__(tag)
        self.model_description = None
        self.import_all = import_all
        input = None
        fmi_call_logger = None
        start_values = {}
        validate = False
        step_size = None
        output_interval = None
        start_values = start_values
        apply_default_start_values = False
        input = input
        namespace_ = "t1"
        debug_logging = False
        visible = False
        self.run_after_solve = ['fmi2Terminate_', 'fmi2FreeInstance_']
        # self.post_step = ['completedIntegratorStep_']
        self.fmu_input = pandas.read_csv(fmu_in) if fmu_in is not None else None
        self.dataframe_aliases = {}
        model_description = read_model_description(fmu_filename, validate=validate)
        self.model_description = model_description
        self.value_ref_used = self.set_variables(self.model_description)
        required_paths = ['resources', 'binaries/']
        tempdir = extract(fmu_filename, include=lambda n: n.startswith(tuple(required_paths)))
        unzipdir = tempdir
        fmi_type = "ModelExchange"
        experiment = model_description.defaultExperiment
        start_time = 0.0
        start_time = float(start_time)

        stop_time = start_time + 0.1

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
        self.fmu = fmu
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

        fmu.setupExperiment(startTime=start_time, stopTime=stop_time)

        input = Input(fmu, model_description, input)

        apply_start_values(fmu, model_description, start_values, apply_default_start_values)

        fmu.enterInitializationMode()
        input.apply(start_time)
        fmu.exitInitializationMode()

        component = fmu.component

        getreal, set_time, fmi2SetC, fmi2SetReal, completedIntegratorStep, \
        get_event_indicators, enter_event_mode, \
        enter_cont_mode, newDiscreteStates = get_fmu_functions(fmu, fmu_logging=fmu_logging)

        fmi2Terminate = getattr(fmu.dll, "fmi2Terminate")
        fmi2Terminate.argtypes = [ctypes.c_void_p]
        fmi2Terminate.restype = ctypes.c_uint

        fmi2FreeInstance = getattr(fmu.dll, "fmi2FreeInstance")
        fmi2FreeInstance.argtypes = [ctypes.c_void_p]
        fmi2FreeInstance.restype = ctypes.c_uint

        @njit()
        def completedIntegratorStep_():
            completedIntegratorStep(component, 1, 0, 0)

        def fmi2Terminate_():
            fmi2Terminate(component)

        def fmi2FreeInstance_():
            fmi2FreeInstance(component)

        self.fmi2Terminate_ = fmi2Terminate_
        self.completedIntegratorStep_ = completedIntegratorStep_
        self.fmi2FreeInstance_ = fmi2FreeInstance_

        len_q = len(self.value_ref_used)
        var_order = [x.valueReference for x in model_description.modelVariables
                     if x.valueReference in self.value_ref_used and (x.type != "String" and x.type != "Boolean")]

        term_1 = np.array([0], dtype=np.int32)
        term_1_ptr = term_1.ctypes.data
        event_1 = np.array([0], dtype=np.int32)
        event_1_ptr = event_1.ctypes.data
        var_array = []
        ptr_var_array = []
        idx_tuple_array = []
        ptr_tuple_array = []
        for i in range(len_q):
            a = np.array([0], dtype=np.float64)
            var_array.append(a)
            ptr_var_array.append(a.ctypes.data)
            idx_tuple_array.append(("a_i_" + str(i), 'a' + str(i)))
            ptr_tuple_array.append(("a" + str(i) + "_ptr", 'a' + str(i)))
        deriv_idx = []
        states_idx = []
        var_names_ordered = []
        var_states_ordered = []
        var_names_ordered_ns = []
        deriv_names_ordered = []
        input_idx = []
        input_var_names_ordered = []
        output_idx = []
        output_var_names_ordered = []
        for idx, variable in enumerate(model_description.modelVariables):
            if variable.valueReference not in self.value_ref_used:
                continue
            if variable.type == 'String':
                continue
            if variable.type == 'Boolean':
                continue

            if variable.derivative:
                deriv_idx.append(self.value_ref_used.index(variable.valueReference))
                states_idx.append(self.value_ref_used.index(
                    [x.valueReference for index, x in enumerate(model_description.modelVariables) if
                     x.name == variable.derivative.name][0]))
                var_names_ordered_ns.append(namespace_ + "." + _replace_name_str(variable.derivative.name) + "_dot")
                deriv_names_ordered.append(_replace_name_str(variable.derivative.name) + "_dot")
                var_states_ordered.append(namespace_ + "." + _replace_name_str(variable.derivative.name))
            else:
                var_names_ordered_ns.append(namespace_ + "." + _replace_name_str(variable.name))
                var_names_ordered.append(_replace_name_str(variable.name))
            if variable.valueReference in self.input_ref:
                input_idx.append(self.value_ref_used.index(variable.valueReference))
                input_var_names_ordered.append(_replace_name_str(variable.name))
            if variable.valueReference in self.output_ref:
                output_idx.append(self.value_ref_used.index(variable.valueReference))
                output_var_names_ordered.append(_replace_name_str(variable.name))

        states_names_ordered = [x.split(".")[1] for x in var_states_ordered]

        fmu.enterContinuousTimeMode()

        q1, equation_call_wrapper = generate_eval_llvm(idx_tuple_array, [idx_tuple_array[i] for i in deriv_idx],
                                                       states_idx, var_order, output_idx)
        module_func = ast.Module(body=[q1, equation_call_wrapper], type_ignores=[])
        if debug_output:
            print(ast.unparse(module_func))
        code = compile(ast.parse(ast.unparse(module_func)), filename='fmu_eval', mode='exec')

        namespace = {"carray": carray, "cfunc": cfunc, "types": types, "np": np, "len_q": len_q, "getreal": getreal,
                     "component": component, "fmi2SetReal": fmi2SetReal, "set_time": set_time, "fmi2SetC": fmi2SetC,
                     "completedIntegratorStep": completedIntegratorStep}
        exec(code, namespace)
        equation_call = namespace["equation_call"]

        q = generate_fmu_eval(var_names_ordered, ptr_tuple_array,
                              [ptr_tuple_array[i] for i in deriv_idx],
                              output_idx, output_var_names_ordered)
        module_func = ast.Module(body=[q], type_ignores=[])
        if debug_output:
            print(ast.unparse(module_func))
        code = compile(ast.parse(ast.unparse(module_func)), filename='fmu_eval', mode='exec')
        namespace = {"NumerousFunction": NumerousFunction, "carray": carray,
                     "address_as_void_pointer": address_as_void_pointer,
                     "equation_call": equation_call,
                     "event_1_ptr": event_1_ptr,
                     "term_1_ptr": term_1_ptr
                     }

        for i in range(len_q):
            namespace.update({"a" + str(i): var_array[i]})
            namespace.update({"a" + str(i) + "_ptr": ptr_var_array[i]})
        exec(code, namespace)

        event_n = model_description.numberOfEventIndicators
        self.fmu_eval = namespace["fmu_eval"]
        event_cond = []
        event_cond_wrapped = []
        self._c_ptrs_a = []
        for _ in range(event_n):
            self._c_ptrs_a.append(np.zeros(shape=event_n))

        for i in range(event_n):
            q, wrapper = generate_eval_event(range(len_q), len_q, var_order, event_id=i)
            module_func = ast.Module(body=[q, wrapper], type_ignores=[])
            if debug_output:
                print(ast.unparse(module_func))
            code = compile(ast.parse(ast.unparse(module_func)), filename='fmu_eval', mode='exec')
            namespace = {"carray": carray, "event_n": event_n, "cfunc": cfunc, "types": types, "np": np, "len_q": len_q,
                         "getreal": getreal,"completedIntegratorStep": completedIntegratorStep,
                         "component": component, "fmi2SetReal": fmi2SetReal, "set_time": set_time,
                         "get_event_indicators": get_event_indicators}
            exec(code, namespace)

            f1, f2 = generate_njit_event_cond(var_states_ordered, i, var_names_ordered_ns)
            module_func = ast.Module(body=[f1, f2], type_ignores=[])
            if debug_output:
                print(ast.unparse(module_func))
            code = compile(ast.parse(ast.unparse(module_func)), filename='fmu_eval_2', mode='exec')
            namespace = {"carray": carray, "event_n": event_n, "cfunc": cfunc, "types": types, "np": np,
                         "event_ind_call_" + str(i): namespace["event_ind_call_" + str(i)],
                         "c_ptr": self._c_ptrs_a[i].ctypes.data,"completedIntegratorStep": completedIntegratorStep,
                         "component": component, "fmi2SetReal": fmi2SetReal, "set_time": set_time,
                         "njit": njit, "address_as_void_pointer": address_as_void_pointer}
            exec(code, namespace)
            event_cond.append(namespace["event_cond_inner_" + str(i)])
            event_cond_2 = namespace["event_cond_" + str(i)]
            event_cond_2.lines = ast.unparse(ast.Module(body=[f2], type_ignores=[]))
            event_cond_wrapped.append(event_cond_2)

        class EventInfo(ctypes.Structure):

            _fields_ = [('newDiscreteStatesNeeded', ctypes.c_int8),
                        ('terminateSimulation', ctypes.c_int8),
                        ('nominalsOfContinuousStatesChanged', ctypes.c_int8),
                        ('valuesOfContinuousStatesChanged', ctypes.c_int8),
                        ('nextEventTimeDefined', ctypes.c_int8),
                        ('nextEventTime', ctypes.c_double)]

        event_info = EventInfo()
        q_ptr = ctypes.addressof(event_info)

        a, b = generate_action_event(len_q, var_order)
        module_func = ast.Module(body=[a, b], type_ignores=[])
        if debug_output:
            print(ast.unparse(module_func))
        code = compile(ast.parse(ast.unparse(module_func)), filename='fmu_eval', mode='exec')
        namespace = {"carray": carray, "event_n": event_n, "cfunc": cfunc, "types": types, "np": np, "len_q": len_q,
                     "getreal": getreal,
                     "q_a": q_ptr,"completedIntegratorStep": completedIntegratorStep,
                     "component": component, "enter_event_mode": enter_event_mode, "set_time": set_time,
                     "get_event_indicators": get_event_indicators, "newDiscreteStates": newDiscreteStates,
                     "enter_cont_mode": enter_cont_mode, "fmi2SetReal": fmi2SetReal}
        exec(code, namespace)
        event_ind_call = namespace["event_ind_call"]

        ae_vars = []
        ae_ptrs = []

        for i in range(len_q):
            a_e = np.array([0], dtype=np.float64)
            ae_vars.append(a_e)
            ae_ptrs.append(a_e.ctypes.data)

        a1, b1 = generate_event_action(len_q, var_names_ordered_ns)

        module_func = ast.Module(body=[a1, b1], type_ignores=[])
        if debug_output:
            print(ast.unparse(module_func))
        code = compile(ast.parse(ast.unparse(module_func)), filename='fmu_eval', mode='exec')
        namespace = {"carray": carray, "event_n": event_n, "cfunc": cfunc, "types": types, "np": np, "len_q": len_q,

                     "component": component, "enter_event_mode": enter_event_mode, "set_time": set_time,
                     "get_event_indicators": get_event_indicators, "event_ind_call": event_ind_call,
                     "njit": njit,"completedIntegratorStep": completedIntegratorStep,
                     "address_as_void_pointer": address_as_void_pointer}

        for i in range(len_q):
            namespace.update({"a_e_" + str(i): ae_vars[i]})
            namespace.update({"a_e_ptr_" + str(i): ae_ptrs[i]})

        exec(code, namespace)
        event_action = namespace["event_action"]
        event_action_2 = namespace["event_action_2"]
        event_action_2.lines = ast.unparse(ast.Module(body=[b1], type_ignores=[]))
        if len(deriv_names_ordered) > 0:
            gec = generate_eq_call(deriv_names_ordered, var_names_ordered, input_var_names_ordered,
                                   output_var_names_ordered,
                                   [x for x in var_names_ordered if x not in states_names_ordered])
            if debug_output:
                print(ast.unparse(gec))
            code = compile(ast.parse(ast.unparse(gec)), filename='fmu_eval', mode='exec')
            namespace = {"Equation": Equation}
            exec(code, namespace)
            result = namespace["eval"]
            result.lines = ast.unparse(gec)
            result.lineno = ast.parse(ast.unparse(gec)).body[0].lineno
            self.eval = ptypes.MethodType(result, self)
            self.equations.append(self.eval)
        self.t1 = self.create_namespace(namespace_)
        self.t1.add_equations([self])
        for i in range(event_n):
            self.add_event("event_" + str(i), event_cond_wrapped[i], event_action_2,
                           compiled_functions={"event_cond_inner_" + str(i): event_cond[i],
                                               "event_action": event_action}, direction=0)

    def set_variables(self, model_description):
        derivatives = []
        value_ref_used = []
        derivatives_names = []
        states = []
        input_ref = []
        output_ref = []

        for variable in model_description.modelVariables:
            if variable.derivative:
                derivatives.append(variable)
                derivatives_names.append(_replace_name_str(variable.derivative.name))

                states.append(variable.derivative)
        for variable in model_description.modelVariables:
            if variable.type == 'String':
                continue
            if variable.type == 'Boolean':
                continue
            if variable in states:
                if variable.start:
                    start = variable.start
                else:
                    start = 0
                value_ref_used.append(variable.valueReference)
                self.add_state(_replace_name_str(variable.name), float(start), create_derivative=False)
                continue

            if variable in derivatives:
                value_ref_used.append(variable.valueReference)
                self.add_derivative(derivatives_names[derivatives.index(variable)])
                continue
            if variable.initial == 'exact':
                if variable.variability == 'fixed':
                    if variable.start != "DISABLED":
                        if isinstance(variable.start, bool):
                            start = int(variable.start is True)
                        else:
                            start = float(variable.start)
                        if self.import_all or variable.causality == 'input' or variable.causality == 'output':
                            value_ref_used.append(variable.valueReference)
                            self.add_constant(_replace_name_str(variable.name), start)
                    else:
                        if self.import_all or variable.causality == 'input' or variable.causality == 'output':
                            value_ref_used.append(variable.valueReference)
                            self.add_constant(_replace_name_str(variable.name), 0.0)
                if variable.variability == 'discrete':
                    if variable.start == "false":
                        if self.import_all or variable.causality == 'input' or variable.causality == 'output':
                            value_ref_used.append(variable.valueReference)
                            self.add_parameter(_replace_name_str(variable.name), 0.0)
                    if variable.start == "true":
                        if self.import_all or variable.causality == 'input' or variable.causality == 'output':
                            value_ref_used.append(variable.valueReference)
                            self.add_parameter(_replace_name_str(variable.name), 1.0)
                if variable.variability == 'tunable':
                    if self.import_all or variable.causality == 'input' or variable.causality == 'output':
                        value_ref_used.append(variable.valueReference)
                        self.add_parameter(_replace_name_str(variable.name), float(variable.start))
            else:
                if not variable.derivative:
                    if self.import_all or variable.causality == 'input' or variable.causality == 'output':
                        value_ref_used.append(variable.valueReference)
                        if variable.causality == 'input':
                            input_ref.append(variable.valueReference)
                        if variable.causality == 'output':
                            output_ref.append(variable.valueReference)
                        self.add_parameter(_replace_name_str(variable.name), 0.0)

        # TODO another types and tunable
        for variable in model_description.modelVariables:
            if variable.causality == 'input':
                if variable.variability == 'discrete':
                    self.dataframe_aliases[variable.name] = (variable.name, InterpolationType.PIESEWISE)
                else:
                    self.dataframe_aliases[variable.name] = (variable.name, InterpolationType.LINEAR)

        if self.fmu_input is not None:
            data_loader = InMemoryDataLoader(self.fmu_input)
            index_to_timestep_mapping = 'time'
            index_to_timestep_mapping_start = 0
            external_mappings = [(ExternalMappingElement("inmemory",
                                                         index_to_timestep_mapping,
                                                         index_to_timestep_mapping_start,
                                                         1,
                                                         self.dataframe_aliases,
                                                         ))]
            self.external_mappings = ExternalMappingUnpacked(external_mappings, data_loader)
        else:
            self.external_mappings = None
        self.input_ref = input_ref
        self.output_ref = output_ref
        return value_ref_used
