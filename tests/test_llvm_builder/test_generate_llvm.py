from numba import carray
from pytest import approx

from numerous.engine.model.lowering.llvm_builder import LLVMBuilder
import numpy as np
import os

initial_values = np.arange(1, 10)
filename = 'llvm_IR_code.txt'

if os.path.exists(filename):
    os.remove(filename)

eval_llvm_signature = 'void(float64, float64, CPointer(float64), CPointer(float64))'


def eval_llvm(s_x1, s_x2, s_x2_dot, s_x3_dot):
    carray(s_x2_dot, (1,))[0] = -100 if s_x1 > s_x2 else 50
    carray(s_x3_dot, (1,))[0] = -carray(s_x2_dot, (1,))[0]


number_of_derivatives = 3
number_of_states = 3

variable_names = {
    "oscillator1.mechanics.x": 0,
    "oscillator1.mechanics.y": 1,
    "oscillator1.mechanics.z": 2,
    "oscillator1.mechanics.a": 3,
    "oscillator1.mechanics.b": 4,
    "oscillator1.mechanics.c": 5,
    "oscillator1.mechanics.x_dot": 6,
    "oscillator1.mechanics.y_dot": 7,
    "oscillator1.mechanics.z_dot": 8,
}


def test_llvm_1_to_1_mapping_state():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)

    llvm_program.add_mapping(["oscillator1.mechanics.x"], ["oscillator1.mechanics.x_dot"])

    diff, var_func, _ = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([2.1, 8, 9.])


def test_llvm_1_to_1_mapping_parameter():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)

    llvm_program.add_mapping(["oscillator1.mechanics.b"], ["oscillator1.mechanics.x_dot"])

    diff, var_func, _ = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([5, 8, 9.])


def test_llvm_n_to_1_sum_mapping():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)

    llvm_program.add_mapping(["oscillator1.mechanics.x", "oscillator1.mechanics.y", "oscillator1.mechanics.b"],
                             ["oscillator1.mechanics.x_dot"])

    diff, var_func, _ = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([9.3, 8, 9.])


def test_llvm_1_to_n_mapping():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)

    llvm_program.add_mapping(["oscillator1.mechanics.x"],
                             ["oscillator1.mechanics.x_dot", "oscillator1.mechanics.y_dot"])

    diff, var_func, _ = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([2.1, 2.1, 9.])


def test_llvm_1_function():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)
    llvm_program.add_external_function(eval_llvm, eval_llvm_signature, 2, 2)

    llvm_program.add_call(eval_llvm.__qualname__,
                          ["oscillator1.mechanics.x", "oscillator1.mechanics.y"],
                          ["oscillator1.mechanics.x_dot", "oscillator1.mechanics.y_dot"])

    diff, var_func, _ = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([50, -50, 9.])
    assert approx(diff(np.array([2.3, 2.2, 2.1]))) == np.array([-100, 100, 9.])


def test_llvm_1_function_and_mapping():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)
    llvm_program.add_external_function(eval_llvm, eval_llvm_signature, 2, 2)

    llvm_program.add_call(eval_llvm.__qualname__,
                          ["oscillator1.mechanics.x", "oscillator1.mechanics.y"],
                          ["oscillator1.mechanics.a", "oscillator1.mechanics.y_dot"])

    llvm_program.add_mapping(args=["oscillator1.mechanics.a"],
                             targets=["oscillator1.mechanics.x_dot"])

    diff, var_func, _ = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([50, -50, 9.])
    assert approx(diff(np.array([2.3, 2.2, 2.1]))) == np.array([-100, 100, 9.])


def test_llvm_1_function_and_mappings():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)
    llvm_program.add_external_function(eval_llvm, eval_llvm_signature, 2, 2)

    llvm_program.add_mapping(args=["oscillator1.mechanics.x"],
                             targets=["oscillator1.mechanics.b"])

    llvm_program.add_call(eval_llvm.__qualname__, ["oscillator1.mechanics.b", "oscillator1.mechanics.y"],
                          ["oscillator1.mechanics.a", "oscillator1.mechanics.y_dot"])

    llvm_program.add_mapping(args=["oscillator1.mechanics.a"],
                             targets=["oscillator1.mechanics.b"])

    llvm_program.add_call(eval_llvm.__qualname__, ["oscillator1.mechanics.b", "oscillator1.mechanics.y"],
                          ["oscillator1.mechanics.a", "oscillator1.mechanics.y_dot"])

    diff, var_func, _ = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([7., 100., 9.])

    assert approx(var_func()) == np.array([2.1, 2.2, 2.3, -100., 50., 6., 7., 100., 9.])


def test_llvm_2_function_and_mappings():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)
    llvm_program.add_external_function(eval_llvm, eval_llvm_signature, 2, 2)

    llvm_program.add_call(eval_llvm.__qualname__, ["oscillator1.mechanics.b", "oscillator1.mechanics.y"],
                          ["oscillator1.mechanics.a", "oscillator1.mechanics.y_dot"])

    diff, var_func, _ = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([7., 100., 9.])

    assert approx(var_func()) == np.array([2.1, 2.2, 2.3, -100., 5., 6., 7., 100., 9.])


def test_llvm_loop():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)
    llvm_program.add_external_function(eval_llvm, eval_llvm_signature, 2, 2)

    llvm_program.add_set_call(eval_llvm.__qualname__, [["oscillator1.mechanics.x", "oscillator1.mechanics.y"],
                                          ["oscillator1.mechanics.z", "oscillator1.mechanics.a"]],
                              [["oscillator1.mechanics.c", "oscillator1.mechanics.x_dot"],
                               ["oscillator1.mechanics.y_dot", "oscillator1.mechanics.z_dot"]])

    diff, var_func, _ = llvm_program.generate(filename)

    assert approx(diff(np.array([2.6, 2.2, 2.3]))) == np.array([100, 50., -50])

    assert approx(var_func()) == np.array([2.6, 2.2, 2.3, 4., 5., -100., 100., 50., -50.])


def test_llvm_idx_write():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)
    llvm_program.add_mapping(["oscillator1.mechanics.b"],
                             ["oscillator1.mechanics.x_dot", "oscillator1.mechanics.y_dot"])
    diff, var_func, var_write = llvm_program.generate(filename)

    var_write(100, 4)

    assert approx(var_func()) == np.array([1.0, 2.0, 3.0, 4., 100., 6., 7., 8., 9.])
    assert approx(diff(np.array([2.6, 2.2, 2.3]))) == np.array([100, 100., 9.])