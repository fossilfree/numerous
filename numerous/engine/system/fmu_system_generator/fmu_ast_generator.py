import ast

# @NumerousFunction()
# all non derivatives in order
# def fmu_eval(e, g, h, v):
# all variables
#     carray(address_as_void_pointer(a0_ptr), a0.shape, dtype=a0.dtype)[0] = 0
#     carray(address_as_void_pointer(a4_ptr), a4.shape, dtype=a0.dtype)[0] = 0
#     carray(address_as_void_pointer(a3_ptr), a3.shape, dtype=a0.dtype)[0] = 0
#     carray(address_as_void_pointer(a2_ptr), a3.shape, dtype=a0.dtype)[0] = 0
#     carray(address_as_void_pointer(a1_ptr), a3.shape, dtype=a0.dtype)[0] = 0
#     carray(address_as_void_pointer(a5_ptr), a3.shape, dtype=a0.dtype)[0] = 0
#     equation_call(address_as_void_pointer(event_1_ptr),
#                   address_as_void_pointer(term_1_ptr),
#                   address_as_void_pointer(a0_ptr),
#                   address_as_void_pointer(a1_ptr),
#                   address_as_void_pointer(a2_ptr),
#                   address_as_void_pointer(a3_ptr),
#                   address_as_void_pointer(a4_ptr),
#                   address_as_void_pointer(a5_ptr), h, v, g, e,
#                   0.1)
# # all derivatives in order
#     return carray(address_as_void_pointer(a1_ptr), (1,), dtype=np.float64)[0], \
#            carray(address_as_void_pointer(a3_ptr), (1,), dtype=np.float64)[0]


CARRAY = 'carray'
DTYPE = 'dtype'
SHAPE = 'shape'
ADDRESS_FUNC = 'address_as_void_pointer'
EQ_CALL = 'equation_call'
EVENT_1_PTR = "event_1_ptr"
TERM_1_PTR = "term_1_ptr"
NUMEROUS_FUNCTION= 'NumerousFunction'


def generate_fmu_eval(input_args, zero_assign_ptrs, output_ptrs):
    args_lst = []
    for arg_id in input_args:
        args_lst.append(ast.arg(arg=arg_id))
    args = ast.arguments(posonlyargs=[], args=args_lst, kwonlyargs=[],
                         kw_defaults=[], defaults=[])

    zero_assigns = []
    for zero_assign_ptr, arr_id in zero_assign_ptrs:
        shape = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=SHAPE, ctx=ast.Load())
        dtype_kw = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=DTYPE, ctx=ast.Load())
        _keywords = ast.keyword(arg=DTYPE, value=dtype_kw)
        zero_assigns.append(ast.Assign(targets=[carray_call(add_address_as_void_pointer(zero_assign_ptr),
                                                            shape, _keywords)],
                                       value=ast.Constant(value=0), lineno=0))
    eq_call_args = [add_address_as_void_pointer(EVENT_1_PTR), add_address_as_void_pointer(TERM_1_PTR)]

    for zero_assign_ptr, _ in zero_assign_ptrs:
        eq_call_args.append(add_address_as_void_pointer(zero_assign_ptr))
    for arg_id in input_args:
        eq_call_args.append(ast.Name(id=arg_id, ctx=ast.Load()))

    eq_call_args.append(ast.Constant(value=0.1))
    eq_expr = [ast.Expr(value=ast.Call(func=ast.Name(id=EQ_CALL), args=eq_call_args,  keywords=[], ctx=ast.Load()),  keywords=[])]
    return_elts = []
    for zero_assign_ptr, arr_id in output_ptrs:
        shape = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=SHAPE, ctx=ast.Load())
        dtype_kw = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=DTYPE, ctx=ast.Load())
        _keywords = ast.keyword(arg=DTYPE, value=dtype_kw)
        return_elts.append(carray_call(add_address_as_void_pointer(zero_assign_ptr), shape, _keywords))
    return_exp = [ast.Return(value=ast.Tuple(elts=return_elts),ctx=ast.Load())]
    decorator_list = [ast.Call(func=ast.Name(id=NUMEROUS_FUNCTION, ctx=ast.Load()), args=[], keywords=[])]
    return ast.FunctionDef(name='fmu_eval', args=args,body = zero_assigns+eq_expr+return_exp, decorator_list=decorator_list,lineno=0)


def dtype(id: str):
    return ast.keyword(arg=DTYPE,
                       value=ast.Attribute(value=ast.Name(id=id, ctx=ast.Load()), attr=DTYPE, ctx=ast.Load()))


def carray_call(ptr_arg: ast.expr, shape: ast.Attribute, dtype_kw: ast.keyword):
    return ast.Subscript(value=ast.Call(func=ast.Name(id=CARRAY, ctx=ast.Load()), args=[ptr_arg, shape],
                         keywords=[dtype_kw]), slice=ast.Constant(value=0), ctx=ast.Store())


def add_address_as_void_pointer(var_id):
    return ast.Call(func=ast.Name(id=ADDRESS_FUNC, ctx=ast.Load()),  keywords=[], args=[ast.Name(id=var_id,
                                                                                   ctx=ast.Load())])
