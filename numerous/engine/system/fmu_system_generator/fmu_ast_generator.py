import ast

CARRAY = 'carray'
DTYPE = 'dtype'
SHAPE = 'shape'
ADDRESS_FUNC = 'address_as_void_pointer'
EQ_CALL = 'equation_call'
EVENT_1_PTR = "event_1_ptr"
TERM_1_PTR = "term_1_ptr"
NUMEROUS_FUNCTION = 'NumerousFunction'


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
    eq_expr = [ast.Expr(value=ast.Call(func=ast.Name(id=EQ_CALL), args=eq_call_args, keywords=[], ctx=ast.Load()),
                        keywords=[])]
    return_elts = []
    for zero_assign_ptr, arr_id in output_ptrs:
        shape = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=SHAPE, ctx=ast.Load())
        dtype_kw = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=DTYPE, ctx=ast.Load())
        _keywords = ast.keyword(arg=DTYPE, value=dtype_kw)
        return_elts.append(carray_call(add_address_as_void_pointer(zero_assign_ptr), shape, _keywords))
    return_exp = [ast.Return(value=ast.Tuple(elts=return_elts), ctx=ast.Load())]
    decorator_list = [ast.Call(func=ast.Name(id=NUMEROUS_FUNCTION, ctx=ast.Load()), args=[], keywords=[])]
    return ast.FunctionDef(name='fmu_eval', args=args, body=zero_assigns + eq_expr + return_exp,
                           decorator_list=decorator_list, lineno=0)


def dtype(id: str):
    return ast.keyword(arg=DTYPE,
                       value=ast.Attribute(value=ast.Name(id=id, ctx=ast.Load()), attr=DTYPE, ctx=ast.Load()))


def carray_call(ptr_arg: ast.expr, shape: ast.Attribute, dtype_kw: ast.keyword):
    return ast.Subscript(value=ast.Call(func=ast.Name(id=CARRAY, ctx=ast.Load()), args=[ptr_arg, shape],
                                        keywords=[dtype_kw]), slice=ast.Constant(value=0), ctx=ast.Store())


def add_address_as_void_pointer(var_id):
    return ast.Call(func=ast.Name(id=ADDRESS_FUNC, ctx=ast.Load()), keywords=[], args=[ast.Name(id=var_id,
                                                                                                ctx=ast.Load())])


# def eval_llvm(event, term, a0, a1, a2, a3, a4, a5, a_i_0, a_i_2, a_i_4, a_i_5, t):
#     vr = np.arange(0, len_q, 1, dtype=np.uint32)
#     value = np.zeros(len_q, dtype=np.float64)
#     ## we are reading derivatives from FMI
#     getreal(component, vr.ctypes, len_q, value.ctypes)
#     value1 = np.array([a_i_0, value[1], a_i_2, value[3], a_i_4, a_i_5], dtype=np.float64)
#     fmi2SetReal(component, vr.ctypes, len_q, value1.ctypes)
#     set_time(component, t)
#     completedIntegratorStep(component, 1, event, term)
#     getreal(component, vr.ctypes, len_q, value.ctypes)
#     carray(a0, (1,), dtype=np.float64)[0] = value[0]
#     carray(a1, (1,), dtype=np.float64)[0] = value[1]
#     carray(a2, (1,), dtype=np.float64)[0] = value[2]
#     carray(a3, (1,), dtype=np.float64)[0] = value[3]
#     carray(a4, (1,), dtype=np.float64)[0] = value[4]
#     carray(a5, (1,), dtype=np.float64)[0] = value[5]
#
# Module(body=[FunctionDef(name='eval_llvm', args=arguments(posonlyargs=[], args=[arg(arg='event'), arg(arg='term'), arg(arg='a0'), arg(arg='a1'), arg(arg='a2'), arg(arg='a3'), arg(arg='a4'), arg(arg='a5'), arg(arg='a_i_0'), arg(arg='a_i_2'), arg(arg='a_i_4'), arg(arg='a_i_5'), arg(arg='t')], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[Assign(targets=[Name(id='vr', ctx=Store())], value=Call(func=Attribute(value=Name(id='np', ctx=Load()), attr='arange', ctx=Load()), args=[Constant(value=0), Name(id='len_q', ctx=Load()), Constant(value=1)], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='uint32', ctx=Load()))])), Assign(targets=[Name(id='value', ctx=Store())], value=Call(func=Attribute(value=Name(id='np', ctx=Load()), attr='zeros', ctx=Load()), args=[Name(id='len_q', ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))])),
# , Assign(targets=[Name(id='value1', ctx=Store())], value=Call(func=Attribute(value=Name(id='np', ctx=Load()), attr='array', ctx=Load()), args=[List(elts=[Name(id='a_i_0', ctx=Load()), Subscript(value=Name(id='value', ctx=Load()), slice=Constant(value=1), ctx=Load()), Name(id='a_i_2', ctx=Load()), Subscript(value=Name(id='value', ctx=Load()), slice=Constant(value=3), ctx=Load()), Name(id='a_i_4', ctx=Load()), Name(id='a_i_5', ctx=Load())], ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))])), Expr(value=Call(func=Name(id='fmi2SetReal', ctx=Load()), args=[Name(id='component', ctx=Load()), Attribute(value=Name(id='vr', ctx=Load()), attr='ctypes', ctx=Load()), Name(id='len_q', ctx=Load()), Attribute(value=Name(id='value1', ctx=Load()), attr='ctypes', ctx=Load())], keywords=[])),
# Expr(value=Call(func=Name(id='set_time', ctx=Load()), args=[Name(id='component', ctx=Load()), Name(id='t', ctx=Load())], keywords=[])),
# Expr(value=Call(func=Name(id='getreal', ctx=Load()), args=[Name(id='component', ctx=Load()), Attribute(value=Name(id='vr', ctx=Load()), attr='ctypes', ctx=Load()), Name(id='len_q', ctx=Load()), Attribute(value=Name(id='value', ctx=Load()), attr='ctypes', ctx=Load())], keywords=[])), Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Name(id='a0', ctx=Load()), Tuple(elts=[Constant(value=1)], ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Subscript(value=Name(id='value', ctx=Load()), slice=Constant(value=0), ctx=Load())), Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Name(id='a1', ctx=Load()), Tuple(elts=[Constant(value=1)], ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Subscript(value=Name(id='value', ctx=Load()), slice=Constant(value=1), ctx=Load())), Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Name(id='a2', ctx=Load()), Tuple(elts=[Constant(value=1)], ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Subscript(value=Name(id='value', ctx=Load()), slice=Constant(value=2), ctx=Load())), Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Name(id='a3', ctx=Load()), Tuple(elts=[Constant(value=1)], ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Subscript(value=Name(id='value', ctx=Load()), slice=Constant(value=3), ctx=Load())),
#
#
# Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Name(id='a4', ctx=Load()), Tuple(elts=[Constant(value=1)], ctx=Load())],
# keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))]), slice=Constant(value=0), ctx=Store())],
# value=Subscript(value=Name(id='value', ctx=Load()), slice=Constant(value=4), ctx=Load())),
#
#
# Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Name(id='a5', ctx=Load()), Tuple(elts=[Constant(value=1)], ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Subscript(value=Name(id='value', ctx=Load()), slice=Constant(value=5), ctx=Load()))], decorator_list=[])], type_ignores=[])


def generate_eval_llvm(assign_ptrs, output_args):
    args_lst = []
    args_lst.append(ast.arg(arg="event"))
    args_lst.append(ast.arg(arg="term"))
    output_args_as_list = [item for t in output_args for item in t]
    for assign_ptr, arr_id in assign_ptrs:
        args_lst.append(ast.arg(arg=arr_id))
    for assign_ptr, arr_id in assign_ptrs:
        if assign_ptr in output_args_as_list:
            continue
        args_lst.append(ast.arg(arg=assign_ptr))
    args_lst.append(ast.arg(arg="t"))
    args = ast.arguments(posonlyargs=[], args=args_lst, kwonlyargs=[],
                         kw_defaults=[], defaults=[])
    body = []
    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='set_time', ctx=ast.Load()),
                    args=[ast.Name(id='component', ctx=ast.Load()), ast.Name(id='t', ctx=ast.Load())], keywords=[])))
    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='completedIntegratorStep', ctx=ast.Load()),
         args=[ast.Name(id='component', ctx=ast.Load()), ast.Constant(value=1), ast.Name(id='event', ctx=ast.Load()),
               ast.Name(id='term', ctx=ast.Load())], keywords=[]),
         keywords=[]))
    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='getreal', ctx=ast.Load()),
         args=[ast.Name(id='component', ctx=ast.Load()), ast.Attribute(value=ast.Name(id='vr', ctx=ast.Load()), attr='ctypes', ctx=ast.Load()),
               ast.Name(id='len_q', ctx=ast.Load()), ast.Attribute(value=ast.Name(id='value', ctx=ast.Load()), attr='ctypes', ctx=ast.Load())],
         keywords=[]),
         keywords=[]))
    return_elts = []
    for idx,(assign_ptr, arr_id) in enumerate(assign_ptrs):
        arr_id_n = ast.Name(id=arr_id, ctx=ast.Load())
        shape = ast.Attribute(value=arr_id_n, attr=SHAPE, ctx=ast.Load())
        dtype_kw = ast.Attribute(value=arr_id_n, attr=DTYPE, ctx=ast.Load())
        _keywords = ast.keyword(arg=DTYPE, value=dtype_kw)
        value = ast.Subscript(value=ast.Name(id='value', ctx=ast.Load()), slice=ast.Constant(value=idx), ctx=ast.Load())
        return_elts.append(ast.Assign(targets=[carray_call(arr_id_n,
                                                           shape, _keywords)],
                                      value=value, lineno=0))
    return ast.FunctionDef(name='eval_llvm', args=args, body=body+return_elts, decorator_list=[], lineno=0)

# generate_fmu_eval(['h', 'v', 'g', 'e'], [('a0_ptr', 'a0'), ('a1_ptr', 'a1'), ('a2_ptr', 'a2'),
#                                                      ('a3_ptr', 'a3'), ('a4_ptr', 'a4'), ('a5_ptr', 'a5')],
#                               [('a1_ptr', 'a1'), ('a3_ptr', 'a3')])
