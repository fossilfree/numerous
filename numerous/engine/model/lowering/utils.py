from enum import IntEnum, unique
from collections import namedtuple
import ast

from numerous.utils import logger as log


@unique
class VarTypes(IntEnum):
    CONSTANT = 0
    PARAMETER = 1
    DERIV = 2
    STATE = 3
    LOCAL = 4
    TMP = 5


VariableArgument = namedtuple('VariableArgument', 'name is_global_var')


def wrap_module(body):
    mod = ast.Module()
    for i in body:
        if not hasattr(i, "_fields"):
            setattr(i, "_fields", [])
    mod.body = body
    mod.type_ignores = []
    setattr(mod, "_fields", [])
    return mod


def generate_code_file(mod_body, file, imports, external_functions_source=False, names="#"):
    for (module, name) in imports.as_imports:
        mod_body.insert(0, ast.Import(names=[ast.alias(name=module, asname=name)], level=0))
    for (module, name) in imports.from_imports:
        mod_body.insert(0, ast.ImportFrom(module=module, names=[ast.alias(name=name, asname=None)], level=0))
    if external_functions_source:
        mod_body.insert(0, ast.ImportFrom(module=external_functions_source, names=[ast.alias(name='*', asname=None)],
                                          level=0))

    mod = wrap_module(mod_body)
    log.info('Generating Source')
    source = names + ast.unparse(mod)
    return source


# Define helper functions to check the variables and sub lists
def non_unique_check(listname_, list_):
    if len(list_) > len(set(list_)):
        import collections
        raise ValueError(
            f'Non unique {listname_}: {[item for item, count in collections.Counter(list_).items() if count > 1]}')


def contains_dot(str_list):
    for s in str_list:
        if '.' in s:
            raise ValueError(f'. in {s}')


def are_all_scalars(str_list, scalar_variables):
    for s in str_list:
        if not s in scalar_variables:
            raise ValueError(f'Not a scalar: {s}')


def are_all_set_variables(str_list, set_variables):
    for s in str_list:
        if not s in set_variables:
            raise ValueError(f'Not a set variable: {s}')


class Vardef:
    def __init__(self, eq_key, llvm=True, exclude_global_var=False):
        self.vars_inds_map = []
        self.targets = []
        self.exclude_global_var = exclude_global_var
        self.eq_key = eq_key
        self.args = []
        self.llvm = llvm
        self._args_order = []
        self._trgs_order = []

    def format(self, var, read=True):
        if read:
            _ctx = ast.Load()
        else:
            _ctx = ast.Store()
        return ast.Name(id=var.replace('scope.', 's_'), lineno=0, col_offset=0, ctx=_ctx)

    def format_target(self, var, read):
        if read:
            _ctx = ast.Load()
        else:
            _ctx = ast.Store()
        if self.llvm:
            return ast.Subscript(
                slice=ast.Index(value=ast.Constant(value=0, lineno=0, col_offset=0), lineno=0, col_offset=0),
                value=ast.Call(
                    args=[ast.Name(id=var.replace('scope.', 's_'), lineno=0, col_offset=0, ctx=ast.Load()),
                          ast.Tuple(ctx=ast.Load(), elts=[ast.Constant(value=1, lineno=0, col_offset=0)], lineno=0,
                                    col_offset=0)],
                    func=ast.Name(id='carray', lineno=0, col_offset=0, ctx=ast.Load()),
                    keywords=[], lineno=0, col_offset=0), lineno=0, col_offset=0, ctx=_ctx)
        else:
            return ast.Name(id=var.replace('scope.', 's_'), lineno=0, col_offset=0, ctx=_ctx)

    def order_variables(self, order_data):
        for sv in order_data:
            if self.eq_key in sv.eq_used:
                self._args_order.append(VariableArgument("scope." + sv.tag, sv.global_var))
            else:
                self._args_order.append(VariableArgument(sv.id, sv.global_var))

        for sv in order_data:
            tmp_v = "scope." + sv.tag
            if tmp_v in self.targets:
                self._trgs_order.append(tmp_v)

    def var_def(self, var, ctxread, read=True):
        if not var in self.vars_inds_map:
            self.vars_inds_map.append(var)
        if read and 'scope.' in var:
            if var not in self.targets and var not in self.args:
                self.args.append(var)
        elif 'scope.' in var:

            if var not in self.targets:
                self.targets.append(var)

        if var in self.targets:
            return self.format_target(var, ctxread)
        else:
            return self.format(var, ctxread)

    def get_order_args(self):
        if self.exclude_global_var:
            return [a for a in self._args_order if not a.is_global_var]
        else:
            return [a for a in self._args_order]

    def get_formatted_order_args(self):
        if self.exclude_global_var:
            return [self.format(a[0], False) for a in self._args_order if not a.is_global_var]
        else:
            return [self.format(a[0], False) for a in self._args_order]

    def get_order_trgs(self):
        return self._trgs_order

    def get_formatted_order_trgs(self):
        return [self.format(a, False) for a in self._trgs_order]

    def get_args(self):
        return self.args

    def get_formatted_args(self):
        return [self.format(a, False) for a in self.args]

    def get_targets(self):
        return self.targets

    def get_formatted_targets(self):
        return [self.format(a, False) for a in self.targets]
