import ast

CARRAY = 'carray'
DTYPE = 'dtype'
SHAPE = 'shape'
ADDRESS_FUNC = 'address_as_void_pointer'
EQ_CALL = 'equation_call'
EVENT_1_PTR = "event_1_ptr"
TERM_1_PTR = "term_1_ptr"
NUMEROUS_FUNCTION = 'NumerousFunction'


def generate_fmu_eval(input_args, zero_assign_ptrs, output_ptrs, parameters_return_idx, fmu_output_args):
    args_lst = []
    for arg_id in input_args:
        if arg_id not in fmu_output_args:
            args_lst.append(ast.arg(arg=arg_id))
    args_lst.append(ast.arg(arg="_time"))

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
        if arg_id not in fmu_output_args:
            eq_call_args.append(ast.Name(id=arg_id, ctx=ast.Load()))

    eq_call_args.append(ast.Name(id="_time", ctx=ast.Load()))
    eq_expr = [ast.Expr(value=ast.Call(func=ast.Name(id=EQ_CALL), args=eq_call_args, keywords=[], ctx=ast.Load()),
                        keywords=[])]
    return_elts = []
    return_exp = []
    for zero_assign_ptr, arr_id in output_ptrs:
        shape = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=SHAPE, ctx=ast.Load())
        dtype_kw = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=DTYPE, ctx=ast.Load())
        _keywords = ast.keyword(arg=DTYPE, value=dtype_kw)
        return_elts.append(carray_call(add_address_as_void_pointer(zero_assign_ptr), shape, _keywords))
    for ix in parameters_return_idx:
        zero_assign_ptr, arr_id = zero_assign_ptrs[ix]
        shape = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=SHAPE, ctx=ast.Load())
        dtype_kw = ast.Attribute(value=ast.Name(id=arr_id, ctx=ast.Load()), attr=DTYPE, ctx=ast.Load())
        _keywords = ast.keyword(arg=DTYPE, value=dtype_kw)
        return_elts.append(carray_call(add_address_as_void_pointer(zero_assign_ptr), shape, _keywords))
    if len(return_elts) > 1:
        return_exp = [ast.Return(value=ast.Tuple(elts=return_elts), ctx=ast.Load())]
    if len(return_elts) == 1:
        return_exp = [ast.Return(value=return_elts[0], ctx=ast.Load())]

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
                                   args=[ast.Name(id='component', ctx=ast.Load()), ast.Name(id='vr', ctx=ast.Load()),
                                         ast.Name(id='len_q', ctx=ast.Load()), ast.Name(id='value', ctx=ast.Load())],
                                   keywords=[]),
                    keywords=[])


def _generate_pointer(attr):
    return ast.Attribute(value=ast.Name(id='types', ctx=ast.Load()), attr=attr, ctx=ast.Load())


def list_from_var_order(var_order: list):
    elts_ = []
    for i_d in var_order:
        elts_.append(ast.Constant(value=i_d))
    return ast.List(elts_, ctx=ast.Load())


def generate_eval_llvm(assign_ptrs, output_args, states_idx, var_order: list, output_idx):
    args_lst = [ast.arg(arg="event"), ast.arg(arg="term")]
    output_args_as_list = [item for t in output_args for item in t]
    for idx_o in output_idx:
        for x in [item for item in assign_ptrs[idx_o]]:
            output_args_as_list.append(x)
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
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()),
                                          args=[list_from_var_order(var_order)],
                                          keywords=[ast.keyword(arg='dtype',
                                                                value=ast.Attribute(value=ast.Name(id='np',
                                                                                                   ctx=ast.Load()),
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

    body.append(generate_set_fmi_update(arg_elts))
    args_ = []
    for state_idx in states_idx:
        args_.append(ast.Name(id=assign_ptrs[state_idx][0], ctx=ast.Load()))
    if len(args_) > 0:
        body.append(ast.Assign(targets=[ast.Name(id='value3', ctx=ast.Store())],
                               value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                                 ctx=ast.Load()),
                                              args=[ast.List(elts=args_, ctx=ast.Load())],
                                              keywords=[ast.keyword(arg='dtype', value=ast.Attribute(
                                                  value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                                  ctx=ast.Load()))]), lineno=0))

        body.append(ast.Expr(value=ast.Call(func=ast.Name(id='fmi2SetC', ctx=ast.Load()),
                                            args=[ast.Name(id='component', ctx=ast.Load()),
                                                  ast.Name(id='value3', ctx=ast.Load()),
                                                  ast.Constant(value=len(states_idx))], keywords=[])))

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


def generate_eval_event(state_idx, len_q, var_order: list, event_id):
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
                       value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                         ctx=ast.Load()),
                                      args=[list_from_var_order(var_order)],
                                      keywords=[ast.keyword(arg='dtype',
                                                            value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
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
        arg_elts.append(ast.Name(id="y" + str(idx), ctx=ast.Load()))

    body.append(generate_set_fmi_update(arg_elts))
    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='set_time', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Name(id='t', ctx=ast.Load())], keywords=[])))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='get_event_indicators', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Name(id='value_event', ctx=ast.Load()),
                                              ast.Name(id='event_n', ctx=ast.Load())],
                                        keywords=[])))
    lst_arg = []
    for i in range(len_q):
        lst_arg.append(ast.Subscript(value=ast.Name(id='value', ctx=ast.Load()),
                                     slice=ast.Constant(value=i), ctx=ast.Load()))

    body.append(ast.Assign(targets=[ast.Name(id='value2', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()),
                                          args=[ast.List(elts=lst_arg, ctx=ast.Load())],
                                          keywords=[ast.keyword(arg='dtype', value=ast.Attribute(
                                              value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                              ctx=ast.Load()))]), lineno=0))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='fmi2SetReal', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Name(id='vr', ctx=ast.Load()),
                                              ast.Name(id='len_q', ctx=ast.Load()),
                                              ast.Name(id='value2', ctx=ast.Load())], keywords=[])),
                )

    body.append(ast.Assign(targets=[ast.Subscript(
        value=ast.Call(func=ast.Name(id='carray', ctx=ast.Load()),
                       args=[ast.Name(id='event_indicators', ctx=ast.Load()),
                             ast.Tuple(elts=[ast.Constant(value=1)],
                                       ctx=ast.Load())],
                       keywords=[ast.keyword(arg='dtype',
                                             value=ast.Attribute(
                                                 value=ast.Name(id='np',
                                                                ctx=ast.Load()),
                                                 attr='float64',
                                                 ctx=ast.Load()))]),
        slice=ast.Constant(value=0), ctx=ast.Store())],
        value=ast.Subscript(value=ast.Name(id='value_event', ctx=ast.Load()),
                            slice=ast.Constant(value=event_id), ctx=ast.Load()), lineno=0))

    wrapper_args = [_generate_pointer('voidptr'), _generate_pointer('float64')]

    for _ in range(len_q):
        wrapper_args.append(_generate_pointer('float64'))

    wrapper = ast.Assign(targets=[ast.Name(id="event_ind_call_" + str(event_id), ctx=ast.Store())],
                         value=ast.Call(func=ast.Call(func=ast.Name(id='cfunc', ctx=ast.Load()),
                                                      args=[ast.Call(
                                                          func=ast.Attribute(value=ast.Name(id='types', ctx=ast.Load()),
                                                                             attr='void', ctx=ast.Load()),
                                                          args=wrapper_args,
                                                          keywords=[])], keywords=[]),
                                        args=[ast.Name(id='eval_event', ctx=ast.Load())],
                                        keywords=[]), lineno=0)

    return ast.FunctionDef(name='eval_event', args=args, body=body, decorator_list=[], lineno=0), wrapper


def generate_njit_event_cond(states, id_, variables):
    body = [ast.Assign(targets=[ast.Name(id='temp_addr', ctx=ast.Store())],
                       value=ast.Call(func=ast.Name(id='address_as_void_pointer', ctx=ast.Load()),
                                      args=[ast.Name(id='c_ptr', ctx=ast.Load())],
                                      keywords=[]), lineno=0),
            ast.Assign(targets=[ast.Subscript(value=ast.Call(func=ast.Name(id='carray', ctx=ast.Load()),
                                                             args=[ast.Name(id='temp_addr', ctx=ast.Load()),
                                                                   ast.Tuple(elts=[ast.Constant(value=1)],
                                                                             ctx=ast.Load())],
                                                             keywords=[ast.keyword(arg='dtype',
                                                                                   value=ast.Attribute(
                                                                                       value=ast.Name(id='np',
                                                                                                      ctx=ast.Load()),
                                                                                       attr='float64',
                                                                                       ctx=ast.Load()))]),
                                              slice=ast.Constant(value=0), ctx=ast.Store())],
                       value=ast.Constant(value=0), lineno=0)]

    args = [ast.Name(id='temp_addr', ctx=ast.Load()), ast.Name(id='t', ctx=ast.Load())]

    for idx, _ in enumerate(variables):
        args.append(
            ast.Subscript(value=ast.Name(id='y', ctx=ast.Load()), slice=ast.Constant(value=idx), ctx=ast.Load()))

    body.append(ast.Expr(
        value=ast.Call(func=ast.Name(id='event_ind_call_' + str(id_), ctx=ast.Load()),
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

    event_cond = ast.FunctionDef(name='event_cond_inner_' + str(id_),
                                 args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='y')],
                                                    kwonlyargs=[],
                                                    kw_defaults=[], defaults=[]), body=body,
                                 decorator_list=[ast.Name(id='njit', ctx=ast.Load())], lineno=0)

    body = []

    elts = []

    for var in variables:
        elts.append(ast.Subscript(value=ast.Name(id='variables', ctx=ast.Load()),
                                  slice=ast.Constant(value=var), ctx=ast.Load()))

    body.append(ast.Assign(targets=[ast.Name(id='q', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()),
                                          args=[ast.List(elts=elts,
                                                         ctx=ast.Load())], keywords=[]), lineno=0))

    body.append(ast.Return(
        value=ast.Call(func=ast.Name(id='event_cond_inner_' + str(id_), ctx=ast.Load()),
                       args=[ast.Name(id='t', ctx=ast.Load()), ast.Name(id='q', ctx=ast.Load())], keywords=[])))

    event_cond_2 = ast.FunctionDef(name='event_cond_' + str(id_),
                                   args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'), ast.arg(arg='variables')],
                                                      kwonlyargs=[],
                                                      kw_defaults=[], defaults=[]), body=body,
                                   decorator_list=[], lineno=0)

    return event_cond, event_cond_2


def generate_set_fmi_update(arg_elts):
    body = []
    body.append(ast.Assign(targets=[ast.Name(id='value1', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()),
                                          args=[ast.List(elts=arg_elts, ctx=ast.Load())],
                                          keywords=[ast.keyword(arg='dtype', value=ast.Attribute(
                                              value=ast.Name(id='np', ctx=ast.Load()),
                                              attr='float64',
                                              ctx=ast.Load()))]), lineno=0))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='fmi2SetReal', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Name(id='vr', ctx=ast.Load()),
                                              ast.Name(id='len_q', ctx=ast.Load()),
                                              ast.Name(id='value1', ctx=ast.Load())], keywords=[])))
    return body


def generate_action_event(len_q: int, var_order: list):
    args_lst = [ast.arg(arg="t")]
    for state_id in range(len_q):
        args_lst.append(ast.arg(arg="y" + str(state_id)))
    for id_ in range(len_q):
        args_lst.append(ast.arg(arg="a_" + str(id_)))

    args = ast.arguments(posonlyargs=[], args=args_lst, kwonlyargs=[],
                         kw_defaults=[], defaults=[])

    body = []
    arg_elts = []

    for idx in range(len_q):
        arg_elts.append(ast.Name(id="y" + str(idx), ctx=ast.Load()))

    body.append(ast.Assign(targets=[ast.Name(id='vr', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array',
                                                             ctx=ast.Load()),
                                          args=[list_from_var_order(var_order)],
                                          keywords=[ast.keyword(arg='dtype',
                                                                value=ast.Attribute(
                                                                    value=ast.Name(id='np', ctx=ast.Load()),
                                                                    attr='uint32', ctx=ast.Load()))]),
                           lineno=0))
    body.append(generate_set_fmi_update(arg_elts))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='enter_event_mode', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load())], keywords=[])))
    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='newDiscreteStates', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load()),
                                              ast.Name(id='q_a', ctx=ast.Load())], keywords=[])))
    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='enter_cont_mode', ctx=ast.Load()),
                                        args=[ast.Name(id='component', ctx=ast.Load())], keywords=[])))
    body.append(ast.Assign(targets=[ast.Name(id='vr', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                                             attr='array', ctx=ast.Load()),
                                          args=[list_from_var_order(var_order)], keywords=[ast.keyword(arg='dtype',
                                                                                                       value=ast.Attribute(
                                                                                                           value=ast.Name(
                                                                                                               id='np',
                                                                                                               ctx=ast.Load()),
                                                                                                           attr='uint32',
                                                                                                           ctx=ast.Load()))]),
                           lineno=0))
    body.append(ast.Assign(targets=[ast.Name(id='value', ctx=ast.Store())],
                           value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='zeros',
                                                             ctx=ast.Load()),
                                          args=[ast.Name(id='len_q', ctx=ast.Load())],
                                          keywords=[ast.keyword(arg='dtype',
                                                                value=ast.Attribute(
                                                                    value=ast.Name(id='np',
                                                                                   ctx=ast.Load()),
                                                                    attr='float64',
                                                                    ctx=ast.Load()))]), lineno=0))
    body.append(ast.Expr(
        value=ast.Call(func=ast.Name(id='getreal', ctx=ast.Load()), args=[ast.Name(id='component', ctx=ast.Load()),
                                                                          ast.Name(id='vr', ctx=ast.Load()),
                                                                          ast.Name(id='len_q', ctx=ast.Load()),
                                                                          ast.Name(id='value', ctx=ast.Load())],
                       keywords=[])))

    for i in range(len_q):
        body.append(ast.Assign(targets=[ast.Subscript(
            value=ast.Call(func=ast.Name(id='carray', ctx=ast.Load()), args=[ast.Name(id='a_' + str(i), ctx=ast.Load()),
                                                                             ast.Tuple(elts=[ast.Constant(value=1)],
                                                                                       ctx=ast.Load())], keywords=[
                ast.keyword(arg='dtype', value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                                             ctx=ast.Load()))]),
            slice=ast.Constant(value=0), ctx=ast.Store())],
            value=ast.Subscript(value=ast.Name(id='value', ctx=ast.Load()),
                                slice=ast.Constant(value=i), ctx=ast.Load()), lineno=0))

    wrapper_args = [_generate_pointer('float64')]

    for _ in range(len_q):
        wrapper_args.append(_generate_pointer('float64'))

    for _ in range(len_q):
        wrapper_args.append(_generate_pointer('voidptr'))

    wrapper = ast.Assign(targets=[ast.Name(id="event_ind_call", ctx=ast.Store())],
                         value=ast.Call(func=ast.Call(func=ast.Name(id='cfunc', ctx=ast.Load()),
                                                      args=[ast.Call(
                                                          func=ast.Attribute(value=ast.Name(id='types', ctx=ast.Load()),
                                                                             attr='void', ctx=ast.Load()),
                                                          args=wrapper_args,
                                                          keywords=[])], keywords=[]),
                                        args=[ast.Name(id='event_callback_fun', ctx=ast.Load())],
                                        keywords=[]), lineno=0)

    return ast.FunctionDef(name='event_callback_fun', args=args, body=body, decorator_list=[], lineno=0), wrapper


def generate_event_action(len_q, variables):
    body = []
    for i in range(len_q):
        body.append(ast.Assign(targets=[ast.Subscript(
            value=ast.Call(func=ast.Name(id='carray', ctx=ast.Load()),
                           args=[ast.Call(func=ast.Name(id='address_as_void_pointer', ctx=ast.Load()),
                                          args=[ast.Name(id='a_e_ptr_' + str(i), ctx=ast.Load())],
                                          keywords=[]),
                                 ast.Attribute(value=ast.Name(id='a_e_' + str(i), ctx=ast.Load()), attr='shape',
                                               ctx=ast.Load())],
                           keywords=[ast.keyword(arg='dtype',
                                                 value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                                                     attr='float64',
                                                                     ctx=ast.Load()))]),
            slice=ast.Constant(value=0), ctx=ast.Store())], value=ast.Constant(value=0), lineno=0))

    args = [ast.Name(id='t', ctx=ast.Load())]

    for idx, _ in enumerate(variables):
        args.append(
            ast.Subscript(value=ast.Name(id='y', ctx=ast.Load()), slice=ast.Constant(value=idx), ctx=ast.Load()))

    for i in range(len_q):
        args.append(ast.Call(func=ast.Name(id='address_as_void_pointer', ctx=ast.Load()),
                             args=[ast.Name(id='a_e_ptr_' + str(i), ctx=ast.Load())], keywords=[]))

    body.append(ast.Expr(value=ast.Call(func=ast.Name(id='event_ind_call', ctx=ast.Load()), args=args, keywords=[])))

    elts = []
    for i in range(len_q):
        elts.append(ast.Subscript(
            value=ast.Call(func=ast.Name(id='carray', ctx=ast.Load()), args=[
                ast.Call(func=ast.Name(id='address_as_void_pointer', ctx=ast.Load()),
                         args=[ast.Name(id='a_e_ptr_' + str(i), ctx=ast.Load())],
                         keywords=[]), ast.Tuple(elts=[ast.Constant(value=1)], ctx=ast.Load())], keywords=[
                ast.keyword(arg='dtype', value=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64',
                                                             ctx=ast.Load()))]),
            slice=ast.Constant(value=0), ctx=ast.Load()))

    body.append(ast.Return(value=ast.List(elts=elts, ctx=ast.Load())))

    event_action_fun = ast.FunctionDef(name='event_action', args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='t'),
                                                                                                     ast.arg(arg='y')],
                                                                               kwonlyargs=[], kw_defaults=[],
                                                                               defaults=[]),
                                       body=body, decorator_list=[ast.Name(id='njit', ctx=ast.Load())], lineno=0)

    elts2 = []

    for state in variables:
        elts2.append(ast.Subscript(value=ast.Name(id='variables', ctx=ast.Load()),
                                   slice=ast.Constant(value=state), ctx=ast.Load()))
    body2 = [ast.Assign(targets=[ast.Name(id='q', ctx=ast.Store())],
                        value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),
                                                          attr='array', ctx=ast.Load()),
                                       args=[ast.List(elts=elts2, ctx=ast.Load())], keywords=[]), lineno=0),
             ast.Assign(targets=[ast.Name(id='vars', ctx=ast.Store())],
                        value=ast.Call(func=ast.Name(id='event_action', ctx=ast.Load()),
                                       args=[ast.Name(id='t', ctx=ast.Load()), ast.Name(id='q', ctx=ast.Load())],
                                       keywords=[]), lineno=0)]

    for idx, var_id in enumerate(variables):
        body2.append(ast.Assign(targets=[
            ast.Subscript(value=ast.Name(id='variables', ctx=ast.Load()), slice=ast.Constant(value=var_id),
                          ctx=ast.Store())],
            value=ast.Subscript(value=ast.Name(id='vars', ctx=ast.Load()),
                                slice=ast.Constant(value=idx), ctx=ast.Load()), lineno=0))
    event_action_fun_2 = ast.FunctionDef(name='event_action_2', args=ast.arguments(posonlyargs=[],
                                                                                   args=[ast.arg(arg='t'),
                                                                                         ast.arg(arg='variables')],
                                                                                   kwonlyargs=[],
                                                                                   kw_defaults=[],
                                                                                   defaults=[]),
                                         body=body2,
                                         decorator_list=[], lineno=0)
    return event_action_fun, event_action_fun_2


def generate_eq_call(deriv_names, var_names, input_var_names_ordered, output_var_names_ordered, params_names):
    elts = []
    for d_name in deriv_names:
        elts.append(ast.Attribute(value=ast.Name(id='scope', ctx=ast.Load()), attr=d_name, ctx=ast.Store()))
    for d_name in params_names:
        if d_name not in input_var_names_ordered:
            elts.append(ast.Attribute(value=ast.Name(id='scope', ctx=ast.Load()), attr=d_name, ctx=ast.Store()))

    if len(deriv_names) > 1:
        trg = [ast.Tuple(elts=elts, ctx=ast.Store())]
    else:
        trg = [elts[0]]

    args = []
    for v_name in var_names:
        if v_name not in output_var_names_ordered:
            args.append(ast.Attribute(value=ast.Name(id='scope', ctx=ast.Load()), attr=v_name, ctx=ast.Load()))
    args.append(ast.Attribute(value=ast.Name(id='scope', ctx=ast.Load()), attr="global_vars.t", ctx=ast.Load()))

    return ast.Module(body=[ast.FunctionDef(name='eval',
                                            args=ast.arguments(posonlyargs=[],
                                                               args=[ast.arg(arg='self'), ast.arg(arg='scope')],
                                                               kwonlyargs=[],
                                                               kw_defaults=[], defaults=[]),
                                            body=[ast.Assign(targets=trg, value=ast.Call(
                                                func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()),
                                                                   attr='fmu_eval', ctx=ast.Load()),
                                                args=args, keywords=[]), lineno=0)],
                                            decorator_list=[
                                                ast.Call(func=ast.Name(id='Equation', ctx=ast.Load()), args=[],
                                                         keywords=[])], lineno=0)],
                      type_ignores=[])
