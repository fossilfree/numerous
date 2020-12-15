from __future__ import print_function

import itertools
from ctypes import CFUNCTYPE, POINTER, c_double, c_void_p, c_int64
from numba import carray, cfunc, njit
from numerous import config
import faulthandler
import numpy as np
import logging
import ast

class ASTBuilder:
    """
    Building an AST module.
    """

    def __init__(self, initial_values, variable_names, states, derivatives):
        """
        initial_values - initial values of global variables array. Array should be ordered in  such way
         that all the derivatives are located in the tail.
        variable_names - dictionary
        states - states
        derivatives -  derivatives
        """
        pass

    def add_external_function(self, function, signature, number_of_args, target_ids):
        """
        Wrap the function and make it available in the LLVM module
        """
        pass

    def generate(self, filename=None, save_opt=False):
        pass
    def detailed_print(self, *args, sep=' ', end='\n', file=None):
        if config.PRINT_LLVM:
            print(*args, sep, end, file)

    def save_module(self, filename):
        pass

    def load_global_variable(self, variable_name):
        pass
    def load_state_variable(self,idx, state_name):
        pass

    def store_variable(self, variable_name):
        pass

    def add_call(self, external_function_name, args, target_ids):
        pass

    def add_mapping(self, args, targets):
        pass

    def add_set_call(self, external_function_name, variable_name_arg_and_trg, targets_ids):
        pass

    def add_set_mapping(self, args2d, targets2d):
        pass
