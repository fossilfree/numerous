from pytest import approx

from numerous.engine.model.llvm_builder import LLVMBuilder
import numpy as np
import os

initial_values = np.arange(1, 10)
filename = 'llvm_IR_code.txt'

if os.path.exists(filename):
    os.remove(filename)

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

    diff, var_func = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([2.1, 8, 9.])

def test_llvm_1_to_1_mapping_parameter():

    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)

    llvm_program.add_mapping(["oscillator1.mechanics.b"], ["oscillator1.mechanics.x_dot"])

    diff, var_func = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([5, 8, 9.])


def test_llvm_n_to_1_sum_mapping():

    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)

    llvm_program.add_mapping(["oscillator1.mechanics.x", "oscillator1.mechanics.y", "oscillator1.mechanics.b"],
                             ["oscillator1.mechanics.x_dot"])

    diff, var_func = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([9.3, 8, 9.])

def test_llvm_1_to_n_mapping():
    llvm_program = LLVMBuilder(initial_values, variable_names, number_of_states, number_of_derivatives)

    llvm_program.add_mapping(["oscillator1.mechanics.x"],
                             ["oscillator1.mechanics.x_dot", "oscillator1.mechanics.y_dot"])

    diff, var_func = llvm_program.generate(filename)

    assert approx(diff(np.array([2.1, 2.2, 2.3]))) == np.array([2.1, 2.1, 9.])

# def test_llvm_1_function():
#     llvm_program =
#
# def test_llvm_1_function_and_mapping():
#     llvm_program =
#
# def test_llvm_1_loop_operation():
#     llvm_program =
