from enum import IntEnum, unique

import ast


@unique
class VarTypes(IntEnum):
    CONSTANT = 0
    PARAMETER = 1
    DERIV = 2
    STATE = 3
    LOCAL = 4
    TMP = 5


def wrap_module(body):
    mod = ast.AST()
    mod.body = body
    mod.type_ignores=[]
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
    print('Generating Source')
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
    def __init__(self, llvm=True):
        self.vars_inds_map = []
        self.targets = []
        self.args = []
        self.llvm = llvm
        self.args_order = []
        self.trgs_order = []

    def format(self, var, read=True):
        if read:
            _ctx = ast.Load()
        else:
            _ctx = ast.Store()
        return ast.Name(id=var.replace('scope.', 's_'), lineno=0, col_offset=0, ctx=_ctx)

    def format_target(self, var,read):
        if read:
            _ctx = ast.Load()
        else:
            _ctx = ast.Store()
        if self.llvm:
            return ast.Subscript(slice=ast.Index(value=ast.Num(n=0, lineno=0,col_offset=0), lineno=0,col_offset=0), value=ast.Call(
                args=[ast.Name(id=var.replace('scope.', 's_'), lineno=0,col_offset=0, ctx=ast.Load()), ast.Tuple(ctx=ast.Load(),elts=[ast.Num(n=1,lineno=0,col_offset=0)], lineno=0,col_offset=0)],
                func=ast.Name(id='carray', lineno=0,col_offset=0, ctx=ast.Load()),
                keywords=[], lineno=0,col_offset=0), lineno=0,col_offset=0, ctx=_ctx)
        else:
            return ast.Name(id=var.replace('scope.', 's_'), lineno=0,col_offset=0, ctx=_ctx)

    def order_variables(self, order_data):
        for (var, var_id, used) in order_data:
            if used:
                self.args_order.append("scope." + var)
            else:
                self.args_order.append(var_id)

        for (var, var_id, used) in order_data:
            tmp_v = "scope." + var
            if tmp_v in self.targets:
                self.trgs_order.append(tmp_v)

    def var_def(self, var,ctxread, read=True):
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

    def get_order_args(self, form=True):
        if form:
            result = [self.format(a,False) for a in self.args_order]
        else:
            result = self.args
        result_2 = []
        for name in result:
            result_2.append(ast.arg(arg=name.id,  lineno=0, col_offset=0))
        return result_2

    def get_order_trgs(self, form=True):
        if form:
            return [self.format(a,False) for a in self.trgs_order]
        else:
            return self.args

    def get_args(self, form=True):
        if form:
            return [self.format(a,False) for a in self.args]
        else:
            return self.args

    def get_targets(self, form=True):
        if form:
            return [self.format(a,False) for a in self.targets]
        else:
            return self.targets
