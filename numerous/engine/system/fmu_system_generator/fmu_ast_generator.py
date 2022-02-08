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


def _generate_fet_real_expr():
    return ast.Expr(value=ast.Call(func=ast.Name(id='getreal', ctx=ast.Load()),
                                   args=[ast.Name(id='component', ctx=ast.Load()),
                                         ast.Attribute(value=ast.Name(id='vr', ctx=ast.Load()), attr='ctypes',
                                                       ctx=ast.Load()),
                                         ast.Name(id='len_q', ctx=ast.Load()),
                                         ast.Attribute(value=ast.Name(id='value', ctx=ast.Load()), attr='ctypes',
                                                       ctx=ast.Load())],
                                   keywords=[]),
                    keywords=[])


def _generate_pointer(attr):
    return ast.Attribute(value=ast.Name(id='types', ctx=ast.Load()), attr=attr, ctx=ast.Load())


def generate_eval_llvm(assign_ptrs, output_args):
    args_lst = [ast.arg(arg="event"), ast.arg(arg="term")]
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
    body.append(ast.Assign(targets=[ast.Name(id='vr', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='arange',
                                                             ctx=ast.Load()),
                                          args=[ast.Constant(value=0), ast.Name(id='len_q', ctx=ast.Load()),
                                                ast.Constant(value=1)], keywords=[
                                   ast.keyword(arg='dtype', value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                                                                attr='uint32', ctx=ast.Load()))]),
                           lineno=0))
    body.append(ast.Assign(targets=[ast.Name(id='value', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='zeros',
                                                             ctx=ast.Load()),
                                          args=[ast.Name(id='len_q', ctx=ast.Load())],
                                          keywords=[ast.keyword(arg='dtype', value=ast.Attribute(
                                              value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                              ctx=ast.Load()))]), lineno=0))

    body.append(_generate_fet_real_expr())
    arg_elts = []
    for idx, (assign_ptr, arr_id) in enumerate(assign_ptrs):
        if assign_ptr in output_args_as_list:
            arg_elts.append(ast.Subscript(value=ast.Name(id='value', ctx=ast.Load()),
                                          slice=ast.Constant(value=idx), ctx=ast.Load()))
        else:
            arg_elts.append(ast.Name(id=assign_ptr, ctx=ast.Load()))

    body.append(ast.Assign(targets=[ast.Name(id='value1', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()), args=[ast.List(
                               elts=arg_elts, ctx=ast.Load())], keywords=[
                               ast.keyword(arg='dtype',
                                           value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                                               ctx=ast.Load()))]), lineno=0))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='fmi2SetReal', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Attribute(value=ast.Name(id='vr', ctx=ast.Load()),
                                                            attr='ctypes', ctx=ast.Load()),
                                              ast.Name(id='len_q', ctx=ast.Load()),
                                              ast.Attribute(value=ast.Name(id='value1', ctx=ast.Load()),
                                                            attr='ctypes', ctx=ast.Load())], keywords=[])))
    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='set_time', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Name(id='t', ctx=ast.Load())], keywords=[])))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='completedIntegratorStep', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()), ast.Constant(value=1),
                                              ast.Name(id='event', ctx=ast.Load()),
                                              ast.Name(id='term', ctx=ast.Load())], keywords=[]),
                         keywords=[]))
    body.append(_generate_fet_real_expr())
    return_elts = []
    for idx, (assign_ptr, arr_id) in enumerate(assign_ptrs):
        arr_id_n = ast.Name(id=arr_id, ctx=ast.Load())
        shape = ast.Tuple(elts=[ast.Constant(value=1)], ctx=ast.Load())
        dtype_kw = ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64', ctx=ast.Load())
        _keywords = ast.keyword(arg=DTYPE, value=dtype_kw)
        value = ast.Subscript(value=ast.Name(id='value', ctx=ast.Load()), slice=ast.Constant(value=idx), ctx=ast.Load())
        return_elts.append(ast.Assign(targets=[carray_call(arr_id_n,
                                                           shape, _keywords)],
                                      value=value, lineno=0))

    wrapper_args = [_generate_pointer('voidptr'), _generate_pointer('voidptr')]
    for _, arr_id in assign_ptrs:
        wrapper_args.append(_generate_pointer('voidptr'))

    for assign_ptr, arr_id in assign_ptrs:
        if assign_ptr in output_args_as_list:
            continue
        wrapper_args.append(_generate_pointer('float64'))

    wrapper_args.append(_generate_pointer('float64'))

    wrapper = ast.Assign(targets=[ast.Name(id=EQ_CALL, ctx=ast.Store())],
                         value=ast.Call(func=ast.Call(func=ast.Name(id='cfunc', ctx=ast.Load()),
                                                      args=[ast.Call(
                                                          func=ast.Attribute(value=ast.Name(id='types', ctx=ast.Load()),
                                                                             attr='void', ctx=ast.Load()),
                                                          args=wrapper_args,
                                                          keywords=[])], keywords=[]),
                                        args=[ast.Name(id='eval_llvm', ctx=ast.Load())],
                                        keywords=[]), lineno=0)

    return ast.FunctionDef(name='eval_llvm', args=args, body=body + return_elts, decorator_list=[], lineno=0), wrapper


def generate_eval_event(state_idx, len_q):
    args_lst = [ast.arg(arg="event_indicators"), ast.arg(arg="t")]
    for state_id in state_idx:
        args_lst.append(ast.arg(arg="y" + str(state_id)))

    args = ast.arguments(posonlyargs=[], args=args_lst, kwonlyargs=[],
                         kw_defaults=[], defaults=[])
    body = [ast.Assign(targets=[ast.Name(id='value_event', ctx=ast.Store())],
                       value=ast.Call(
                           func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='zeros', ctx=ast.Load()),
                           args=[ast.Name(id='event_n', ctx=ast.Load())],
                           keywords=[ast.keyword(arg='dtype',
                                                 value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                                                     attr='float64', ctx=ast.Load()))]), lineno=0),
            ast.Assign(targets=[ast.Name(id='vr', ctx=ast.Store())],
                       value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='arange',
                                                         ctx=ast.Load()),
                                      args=[ast.Constant(value=0), ast.Name(id='len_q', ctx=ast.Load()),
                                            ast.Constant(value=1)], keywords=[
                               ast.keyword(arg='dtype', value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                                                            attr='uint32', ctx=ast.Load()))]),
                       lineno=0), ast.Assign(targets=[ast.Name(id='value', ctx=ast.Store())],
                                             value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                                                               attr='zeros',
                                                                               ctx=ast.Load()),
                                                            args=[ast.Name(id='len_q', ctx=ast.Load())],
                                                            keywords=[ast.keyword(arg='dtype', value=ast.Attribute(
                                                                value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                                                ctx=ast.Load()))]), lineno=0),
            _generate_fet_real_expr()]

    arg_elts = []
    for idx in range(len_q):
        if idx not in state_idx:
            arg_elts.append(ast.Subscript(value=ast.Name(id='value', ctx=ast.Load()),
                                          slice=ast.Constant(value=idx), ctx=ast.Load()))
        else:
            arg_elts.append(ast.Name(id="y" + str(idx), ctx=ast.Load()))

    body.append(ast.Assign(targets=[ast.Name(id='value1', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()), args=[ast.List(
                               elts=arg_elts, ctx=ast.Load())], keywords=[
                               ast.keyword(arg='dtype',
                                           value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                                               ctx=ast.Load()))]), lineno=0))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='fmi2SetReal', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Attribute(value=ast.Name(id='vr', ctx=ast.Load()),
                                                            attr='ctypes', ctx=ast.Load()),
                                              ast.Name(id='len_q', ctx=ast.Load()),
                                              ast.Attribute(value=ast.Name(id='value1', ctx=ast.Load()),
                                                            attr='ctypes', ctx=ast.Load())], keywords=[])))
    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='set_time', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Name(id='t', ctx=ast.Load())], keywords=[])))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='get_event_indicators', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()), ast.Attribute(
                                            value=ast.Name(id='value_event', ctx=ast.Load()), attr='ctypes',
                                            ctx=ast.Load()), ast.Name(id='event_n', ctx=ast.Load())],
                                        keywords=[])))
    lst_arg = []
    for i in range(len_q):
        lst_arg.append(ast.Subscript(value=ast.Name(id='value', ctx=ast.Load()),
                                     slice=ast.Constant(value=i), ctx=ast.Load()))

    body.append(ast.Assign(targets=[ast.Name(id='value2', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()), args=[ast.List(
                               elts=lst_arg, ctx=ast.Load())],
                                          keywords=[ast.keyword(arg='dtype', value=ast.Attribute(
                                              value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                              ctx=ast.Load()))]), lineno=0))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='fmi2SetReal', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Attribute(value=ast.Name(id='vr', ctx=ast.Load()),
                                                            attr='ctypes', ctx=ast.Load()),
                                              ast.Name(id='len_q', ctx=ast.Load()),
                                              ast.Attribute(value=ast.Name(id='value2', ctx=ast.Load()),
                                                            attr='ctypes', ctx=ast.Load())], keywords=[])),
                )

    body.append(ast.Assign(targets=[ast.Subscript(value=ast.Call(func=ast.Name(id='carray', ctx=ast.Load()),
                                                                 args=[ast.Name(id='event_indicators', ctx=ast.Load()),
                                                                       ast.Tuple(elts=[ast.Constant(value=1)],
                                                                                 ctx=ast.Load())], keywords=[
            ast.keyword(arg='dtype',
                        value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64', ctx=ast.Load()))]),
                                                  slice=ast.Constant(value=0), ctx=ast.Store())],
                           value=ast.Subscript(value=ast.Name(id='value_event', ctx=ast.Load()),
                                               slice=ast.Constant(value=0), ctx=ast.Load()), lineno=0))

    wrapper_args = [_generate_pointer('voidptr'), _generate_pointer('float64')]

    for _ in state_idx:
        wrapper_args.append(_generate_pointer('float64'))

    wrapper = ast.Assign(targets=[ast.Name(id="event_ind_call_1", ctx=ast.Store())],
                         value=ast.Call(func=ast.Call(func=ast.Name(id='cfunc', ctx=ast.Load()),
                                                      args=[ast.Call(
                                                          func=ast.Attribute(value=ast.Name(id='types', ctx=ast.Load()),
                                                                             attr='void', ctx=ast.Load()),
                                                          args=wrapper_args,
                                                          keywords=[])], keywords=[]),
                                        args=[ast.Name(id='eval_event', ctx=ast.Load())],
                                        keywords=[]), lineno=0)

    return ast.FunctionDef(name='eval_event', args=args, body=body, decorator_list=[], lineno=0), wrapper


def generate_njit_event_cond(states):
    body = [ast.Assign(targets=[ast.Name(id='temp_addr', ctx=ast.Store())],
                       value=ast.Call(func=ast.Name(id='address_as_void_pointer', ctx=ast.Load()),
                                      args=[ast.Name(id='c_ptr', ctx=ast.Load())],
                                      keywords=[]), lineno=0),
            ast.Assign(targets=[ast.Subscript(value=ast.Call(func=ast.Name(id='carray', ctx=ast.Load()),
                                                 args=[ast.Name(id='temp_addr', ctx=ast.Load()),
                                                       ast.Tuple(elts=[ast.Constant(value=1)], ctx=ast.Load())],
                                                 keywords=[ast.keyword(arg='dtype',
                                                                   value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                                                                   attr='float64', ctx=ast.Load()))]),
                                      slice=ast.Constant(value=0), ctx=ast.Store())], value=ast.Constant(value=0),lineno=0)]

    args = [ast.Name(id='temp_addr', ctx=ast.Load()), ast.Name(id='t', ctx=ast.Load())]

    for _ in states:
        args.append(ast.Subscript(value=ast.Name(id='y', ctx=ast.Load()), slice=ast.Constant(value=0), ctx=ast.Load()))

    body.append(ast.Expr(
        value=ast.Call(func=ast.Name(id='event_ind_call_1', ctx=ast.Load()),
                       args=args,
                       keywords=[])))

    body.append(ast.Assign(targets=[ast.Name(id='result', ctx=ast.Store())], value=ast.Subscript(
        value=ast.Call(func=ast.Name(id='carray', ctx=ast.Load()),
                       args=[ast.Name(id='temp_addr', ctx=ast.Load()),
                             ast.Tuple(elts=[ast.Constant(value=1)], ctx=ast.Load())],
                       keywords=[ast.keyword(arg='dtype',
                                             value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                                                 attr='float64',
                                                                 ctx=ast.Load()))]), slice=ast.Constant(value=0),
        ctx=ast.Load()), lineno=0))

    body.append(ast.Return(value=ast.Name(id='result', ctx=ast.Load())))

    event_cond = ast.FunctionDef(name='event_cond',
                                 args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='y')],
                                                    kwonlyargs=[],
                                                    kw_defaults=[], defaults=[]), body=body,
                                 decorator_list=[ast.Name(id='njit', ctx=ast.Load())], lineno=0)

    body = []

    elts = []

    for state in states:
        elts.append(ast.Subscript(value=ast.Name(id='variables', ctx=ast.Load()),
                                  slice=ast.Constant(value=state), ctx=ast.Load()))

    body.append(ast.Assign(targets=[ast.Name(id='q', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()),
                                          args=[ast.List(elts=elts,
                                                         ctx=ast.Load())], keywords=[]), lineno=0))

    body.append(ast.Return(
        value=ast.Call(func=ast.Name(id='event_cond', ctx=ast.Load()),
                       args=[ast.Name(id='t', ctx=ast.Load()), ast.Name(id='q', ctx=ast.Load())], keywords=[])))

    event_cond_2 = ast.FunctionDef(name='event_cond_2',
                                   args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='variables')],
                                                      kwonlyargs=[],
                                                      kw_defaults=[], defaults=[]), body=body,
                                   decorator_list=[], lineno=0)

    return event_cond, event_cond_2
