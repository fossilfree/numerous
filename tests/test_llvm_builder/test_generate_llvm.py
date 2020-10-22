from numerous.engine.model.llvm_builder import LLVMBuilder
import numpy as np
input_array = np.arange(1,30)

def test_llvm_1_mapping():
   number_of_derivatives = 10
   llvm_program = LLVMBuilder(input_array,number_of_derivatives)

def test_llvm_1_function():
    llvm_program =

def test_llvm_1_function_and_mapping():
    llvm_program =

def test_llvm_1_loop_operation():
    llvm_program =
