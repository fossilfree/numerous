import ast
import numpy as np
from numerous.engine.model.lowering.ast_builder import ASTBuilder
from numerous.engine.model.utils import Imports

initial_values = np.arange(1, 10, dtype=np.float64)
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
variable_distributed = {
    "oscillator1.mechanics.a": 0,
    "oscillator1.mechanics.x_dot": 1,
    "oscillator1.mechanics.y_dot": 2,
    "oscillator1.mechanics.z_dot": 3,
    "oscillator1.mechanics.x": 4,
    "oscillator1.mechanics.b": 5,
    "oscillator1.mechanics.c": 6,
    "oscillator1.mechanics.y": 7,
    "oscillator1.mechanics.z": 8,
}
DERIVATIVES = ["oscillator1.mechanics.x_dot", "oscillator1.mechanics.y_dot", "oscillator1.mechanics.z_dot"]
STATES = ["oscillator1.mechanics.x", "oscillator1.mechanics.y", "oscillator1.mechanics.z"]
eval_ast_signature = 'void(float64, float64, CPointer(float64), CPointer(float64))'
eval_ast = ast.parse('''def eval_ast(s_x1, s_x2, s_x2_dot, s_x3_dot):
    s_x2_dot = -100 if s_x1 > s_x2 else 50
    s_x3_dot = -s_x2_dot
    return s_x2_dot,s_x3_dot''').body[0]
imports = Imports()
imports.add_as_import("numpy", "np")


def test_ast_set_unset():
    ast_program = ASTBuilder(initial_values, variable_names, STATES, DERIVATIVES)
    ast_program.add_external_function(eval_ast, eval_ast_signature, number_of_args=4, target_ids=[2, 3])

    ast_program.add_conditional_call(eval_ast.name,
                                     ["oscillator1.mechanics.x", "oscillator1.mechanics.y",
                                      "oscillator1.mechanics.x_dot", "oscillator1.mechanics.y_dot"],
                                     [2, 3],
                                     'call_1')
    assert ast_program.get_call_enabled('call_1')
    ast_program.set_call_enabled('call_1', False)
    assert not ast_program.get_call_enabled('call_1')


def test_ast_defaults():
    ast_program = ASTBuilder(initial_values, variable_names, STATES, DERIVATIVES)
    ast_program.add_external_function(eval_ast, eval_ast_signature, number_of_args=4, target_ids=[2, 3])

    ast_program.add_conditional_call(eval_ast.name,
                                     ["oscillator1.mechanics.x", "oscillator1.mechanics.y",
                                      "oscillator1.mechanics.x_dot",
                                      "oscillator1.mechanics.y_dot"],
                                     [2, 3],
                                     'call_1')

    assert 'def global_kernel(states, eval_ast_flag=True):' in ast_program.unparse(imports)
    ast_program.set_call_enabled('call_1', False)
    assert 'def global_kernel(states, eval_ast_flag=False):' in ast_program.unparse(imports)
