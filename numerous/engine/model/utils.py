from enum import IntEnum, unique
import ast, astor

@unique
class NodeTypes(IntEnum):
    OP=0
    ASSIGN = 1
    VAR=2
    DERIV=3
    STATE=4

class dot_dict:
    def __init__(self, **d):
        for k, v in d.items():
            setattr(self, k, v)

def recurse_Attribute(attr, sep='.'):
    if hasattr(attr,'id'):
        return attr.id
    elif isinstance(attr.value,ast.Name):
        return attr.value.id+sep+attr.attr
    elif isinstance(attr.value, ast.Attribute):
        return recurse_Attribute(attr.value)+sep+attr.attr

def wrap_module(body):
    mod = ast.Module()
    mod.body = body
    return mod


def wrap_function(name, body, args, decorators):
    f = ast.FunctionDef(name)
    f.body = body
    f.decorator_list = decorators
    f.args = args
    return f

def generate_code_file(mod_body, file,preamble="from numba import njit, float64\nimport numpy as np\n"):
    mod = wrap_module(mod_body)
    print('Generating Source')
    source = preamble + astor.to_source(mod, indent_with=' ' * 4,
                                                                                       add_line_information=False,
                                                                                       source_generator_class=astor.SourceGenerator)

    with open(file, 'w') as f:
        f.write(source)