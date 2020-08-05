from enum import IntEnum, unique
from numerous.engine.model.graph import Graph
from numerous.engine.model.source_gen import SourceGeneratorNumerous

import ast, astor

class Vardef:
    def __init__(self):
        self.vars_inds_map = []

    def var_def(self, var, read=True):
        if not var in self.vars_inds_map:
            self.vars_inds_map.append(var)
        ix = self.vars_inds_map.index(var)

        return ast.Subscript(slice=ast.Index(value=ast.Num(n=ix)), value=ast.Name(id='l'))

@unique
class NodeTypes(IntEnum):
    OP=0
    ASSIGN = 1
    VAR=2
    DERIV=3
    STATE=4
    EQUATION=5
    SUM=6
    TMP=7

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

def generate_code_file(mod_body, file,preamble="from numba import njit, carray, float64, float32\nimport numpy as np\n"):
    mod = wrap_module(mod_body)
    print('Generating Source')

    source = preamble + astor.to_source(mod, indent_with=' ' * 4, add_line_information=False, source_generator_class=SourceGeneratorNumerous)

    with open(file, 'w') as f:
        f.write(source)

    return source

