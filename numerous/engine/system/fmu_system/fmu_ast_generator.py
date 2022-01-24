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

# Out[2]: "Module(body=[body=[Assign(targets=[

#
#
#
#
# , Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()),
# args=[Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a4_ptr', ctx=Load())], keywords=[]),
# Attribute(value=Name(id='a4', ctx=Load()), attr='shape', ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='a0', ctx=Load()), attr='dtype', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Constant(value=0)), Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a3_ptr', ctx=Load())], keywords=[]), Attribute(value=Name(id='a3', ctx=Load()), attr='shape', ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='a0', ctx=Load()), attr='dtype', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Constant(value=0)), Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a2_ptr', ctx=Load())], keywords=[]), Attribute(value=Name(id='a3', ctx=Load()), attr='shape', ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='a0', ctx=Load()), attr='dtype', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Constant(value=0)), Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a1_ptr', ctx=Load())], keywords=[]), Attribute(value=Name(id='a3', ctx=Load()), attr='shape', ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='a0', ctx=Load()), attr='dtype', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Constant(value=0)), Assign(targets=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a5_ptr', ctx=Load())], keywords=[]), Attribute(value=Name(id='a3', ctx=Load()), attr='shape', ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='a0', ctx=Load()), attr='dtype', ctx=Load()))]), slice=Constant(value=0), ctx=Store())], value=Constant(value=0)), Expr(value=Call(func=Name(id='equation_call', ctx=Load()), args=[Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='event_1_ptr', ctx=Load())], keywords=[]), Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='term_1_ptr', ctx=Load())], keywords=[]), Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a0_ptr', ctx=Load())], keywords=[]), Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a1_ptr', ctx=Load())], keywords=[]), Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a2_ptr', ctx=Load())], keywords=[]), Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a3_ptr', ctx=Load())], keywords=[]), Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a4_ptr', ctx=Load())], keywords=[]), Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a5_ptr', ctx=Load())], keywords=[]), Name(id='h', ctx=Load()), Name(id='v', ctx=Load()), Name(id='g', ctx=Load()), Name(id='e', ctx=Load()), Constant(value=0.1)], keywords=[])), Return(value=Tuple(elts=[Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a1_ptr', ctx=Load())], keywords=[]), Tuple(elts=[Constant(value=1)], ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))]), slice=Constant(value=0), ctx=Load()), Subscript(value=Call(func=Name(id='carray', ctx=Load()), args=[Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a3_ptr', ctx=Load())], keywords=[]), Tuple(elts=[Constant(value=1)], ctx=Load())], keywords=[keyword(arg='dtype', value=Attribute(value=Name(id='np', ctx=Load()), attr='float64', ctx=Load()))]), slice=Constant(value=0), ctx=Load())], ctx=Load()))], decorator_list=[])], type_ignores=[])"
import ast
CARRAY = 'carray'

def add_argument():
    return ast.arguments

# Call(func=Name(id='address_as_void_pointer', ctx=Load()), args=[Name(id='a0_ptr',ctx=Load())
# Attribute(value=Name(id='a0', ctx=Load()), attr='shape', ctx=Load())]
# keyword(arg='dtype', value=Attribute(value=Name(id='a0', ctx=Load()), attr='dtype', ctx=Load()))
def dtype(id:str):
    return ast.keyword()
def carray_call(ptr_arg:ast.expr, shape:ast.Attribute, dtype:ast.keyword):
    return ast.Subscript(value=ast.Call(func=ast.Name(id=CARRAY, ctx=ast.Load())), args=[ptr_arg, shape],
                         keywords=[dtype], slice=ast.Constant(value=0), ctx=ast.Store())

def add_add_addres_as_void_pointer(var_id):
    pass
# FunctionDef(name='fmu_eval', args=arguments(posonlyargs=[], args=[arg(arg='e'), arg(arg='g'), arg(arg='h'), arg(arg='v')], kwonlyargs=[], kw_defaults=[],
# defaults=[])