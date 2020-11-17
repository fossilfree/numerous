import ast

class Vardef:
    def __init__(self):
        self.vars_inds_map = []

    def var_def(self, var, read=True):
        if not var in self.vars_inds_map:
            self.vars_inds_map.append(var)
        ix = self.vars_inds_map.index(var)

        return ast.Subscript(slice=ast.Index(value=ast.Num(n=ix)), value=ast.Name(id='l'))
