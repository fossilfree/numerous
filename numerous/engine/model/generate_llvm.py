from __future__ import print_function

from ctypes import CFUNCTYPE, POINTER, c_double, c_float, c_void_p, c_int32
from numba import carray, cfunc, njit

try:
    from time import perf_counter as time
except ImportError:
    from time import time

#try:
import faulthandler;

faulthandler.enable()
#except ImportError:
#    pass

import numpy as np

import llvmlite.ir as ll
import llvmlite.binding as llvm


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
llmod_ = llvm.parse_assembly("")

target_machine = llvm.Target.from_default_triple().create_target_machine()
print('target data: ', target_machine.target_data)
print('target triple: ', target_machine.triple)

ee = llvm.create_mcjit_compiler(llmod_, target_machine)
#def generate():
def generate(program, functions, variables, variable_values):
    #@njit('void(float32, float32, CPointer(float32))')
    def exp(s_v, s_x, r):
        s_x_dot = s_v + s_x
        s_v_dot = s_x
        e = carray(r, (2,))
        e[0:2] = s_x_dot, s_v_dot

    t1 = time()

    #Define the overall function
    fnty = ll.FunctionType(ll.FloatType().as_pointer(), [
        #ll.ArrayType(ll.FloatType(),10),
                                            ll.FloatType().as_pointer()
                                           ])


    module = ll.Module()
    ext_funcs = {}
    #Wrap the functions and make them available in the module
    for f in functions:
        print('func: ', f['signature'])
        f_c = cfunc(sig=f['signature'])(f['func'])

        if not 'name' in  f:
            f['name'] = f['func'].__qualname__

        name = f['name']

        f_c_sym = llvm.add_symbol(name, f_c.address)

        fnty_c_func = ll.FunctionType(ll.VoidType(),[ll.FloatType() for i in f['args']] + [ll.FloatType().as_pointer() for i in f['targets']])
        fnty_c_func.as_pointer(f_c_sym)
        f_llvm = ll.Function(module, fnty_c_func, name=name)

        ext_funcs[name] = f_llvm



    #Define global variable array

    max_deriv = -1
    max_var = -1
    max_state = -1

    for ix, p in enumerate(program):#+program[14:]:

        if p['func'] == 'load':


            if p['arg'] == 'variables':
                if p['ix']> max_var:
                    max_var = p['ix']

            elif p['arg'] == 'y':
                if p['ix']> max_state:
                    max_state = p['ix']


            else:
                ValueError('Wrong arg: ', p['arg'])



        elif p['func'] == 'store':



            if p['arg'] == 'variables':
                if p['ix']> max_var:
                    max_var = p['ix']

            elif p['arg'] == 'deriv':
                if p['ix']> max_deriv:
                    max_deriv = p['ix']

            else:
                ValueError('Wrong arg: ', p['arg'])
    max_deriv += 1
    max_state += 1
    max_var += 1
    print('max')
    print(max_deriv)
    print(max_var)
    print(max_state)

    func = ll.Function(module, fnty, name="kernel")

    bb_entry = func.append_basic_block()
    bb_loop = func.append_basic_block()
    bb_exit = func.append_basic_block()



    builder = ll.IRBuilder()
    builder.position_at_end(bb_entry)

    builder.branch(bb_loop)
    builder.position_at_end(bb_loop)


    values = {}

    izero = builder.phi(ll.IntType(32))
    izero.add_incoming(ll.Constant(izero.type, 0), bb_entry)


    var_global = ll.GlobalVariable(module, ll.ArrayType(ll.FloatType(), max_var), 'global_var')
    var_global.initializer = ll.Constant(ll.ArrayType(ll.FloatType(), max_var), [0]*max_var)

    ptr_llvm = {}
    val_llvm = {}

    for var, val in zip(variables, variable_values):
        ptr_llvm[var] = ll.GlobalVariable(module, ll.FloatType(), var)
        ptr_llvm[var].initializer = ll.Constant(ll.FloatType(), val)
        val_llvm[var] = builder.load(ptr_llvm[var], var+'_val')

    max_ret = 1000
    var_returns = ll.GlobalVariable(module, ll.ArrayType(ll.FloatType(), max_ret), 'global_returns')
    var_returns.initializer= ll.Constant(ll.ArrayType(ll.FloatType(), max_ret), [0] * max_ret)

    var_deriv = ll.GlobalVariable(module, ll.ArrayType(ll.FloatType(), max_deriv), 'var_deriv')
    var_deriv.initializer = ll.Constant(ll.ArrayType(ll.FloatType(), max_deriv), [0] * max_deriv)



    index0 = builder.phi(ll.IntType(32))
    index0.add_incoming(ll.Constant(index0.type, 0), bb_entry)

    zero = builder.phi(ll.FloatType())
    zero.add_incoming(ll.Constant(zero.type, 0), bb_entry)

    m1 = builder.phi(ll.FloatType())
    m1.add_incoming(ll.Constant(m1.type, -1), bb_entry)

    p1 = builder.phi(ll.FloatType())
    p1.add_incoming(ll.Constant(p1.type, 1), bb_entry)

    #index0 = builder.phi(ll.IntType(32).as_pointer())
    #index0.add_incoming(ll.Constant(index0.type, 0), bb_entry)
    #variables_pointer = builder.gep(variables, [index0])


    #for i, p in enumerate(program):
    #    print(i, ': ', p)

    for ix, p in enumerate(program):#+program[14:]:

        if p['func'] == 'load':


            if p['arg'] == 'variables':
                pass

                #ptr = builder.gep(var_global, [index0, index])
            elif p['arg'] == 'y':
                index = builder.phi(ll.IntType(32))
                index.add_incoming(ll.Constant(index.type, p['ix']), bb_entry)

                ptr = builder.gep(func.args[0], [index])
                value = builder.load(ptr, name=p['var'])
                val_llvm[p['var']] = value


            else:
                ValueError('Wrong arg: ', p['arg'])



        elif p['func'] == 'store':



            if p['arg'] == 'variables':
                pass
                #ptr = builder.gep(var_global, [index0, index])

            elif p['arg'] == 'deriv':
                index = builder.phi(ll.IntType(32))
                index.add_incoming(ll.Constant(index.type, p['ix']), bb_entry)
                ptr = builder.gep(var_deriv, [index0, index])
                builder.store(val_llvm[p['var']], ptr)
            else:
                ValueError('Wrong arg: ', p['arg'])







        elif p['func'] == 'call':
            #arr = builder.phi(ll.ArrayType(ll.FloatType(), 1).as_pointer())

            #print(p['ext_func'])
            #for i, t in enumerate(p['targets']):
                #if not t in values:

                #    values[t] = builder.phi(ll.FloatType(), t)
                #    values[t].add_incoming(ll.Constant(values[t].type, 0), bb_entry)
                    #raise ValueError('AAARG')
            #if not ix in [13, 15]:

            builder.call(ext_funcs[p['ext_func']],[val_llvm[a] for a in p['args']] + [ptr_llvm[t] for t in p['targets']])
            #builder.call(exp, [values[a] for a in p['args']] + [builder.gep(var_returns, [index0, index0])])


            #for i, t in enumerate(p['targets']):
            #    index = builder.phi(ll.IntType(32))
            #    index.add_incoming(ll.Constant(index.type, i), bb_entry)
            #    targ_pointer = builder.gep(var_returns, [index0, index])
            #    val_llvm[t] = builder.load(targ_pointer)






        elif p['func'] == 'sum':
            accum = builder.phi(ll.FloatType())
            accum.add_incoming(ll.Constant(accum.type, 0), bb_entry)
            la = len(p['args'])
            if la == 0:
                pass
            elif la == 1:
                for t in p['targets']:
                    values[t] = val_llvm[p['args'][0]]
                    val_llvm[t] = val_llvm[p['args'][0]]
            elif la == 2:
                for t in p['targets']:
                    values[t] = builder.fadd(val_llvm[p['args'][0]], val_llvm[p['args'][1]])
                    val_llvm[t] = values[t]
            else:
                accum = builder.fadd(val_llvm[p['args'][0]], val_llvm[p['args'][1]])
                for i, a in enumerate(p['args'][2:-1]):
                    accum = builder.fadd(accum, val_llvm[a])

                values[p['targets'][0]] = builder.fadd(accum, val_llvm[p['args'][-1]])
                val_llvm[p['targets'][0]] = values[p['targets'][0]]

                for t in p['targets'][1:]:
                    values[t] = values[p['targets'][0]]
                    val_llvm[t] = values[p['targets'][0]]



            #v_sum = builder.fadd(accum, zero)




        else:
            raise EnvironmentError('Unknown function: ' + str(p['func']))



        #added = builder.add(accum, value)
    #accum.add_incoming(added, bb_loop)

    #indexp1 = builder.add(index, ll.Constant(index.type, 1))
    #index.add_incoming(indexp1, bb_loop)

    #cond = builder.icmp_unsigned('<', indexp1, func.args[1])
    #builder.cbranch(cond, bb_loop, bb_exit)
    builder.branch(bb_exit)

    builder.position_at_end(bb_exit)
    #builder.ret_void()
    builder.ret(builder.gep(var_deriv, [index0, index0]))

    # Vars function
    fnty_vars = ll.FunctionType(ll.FloatType().as_pointer(), [ll.IntType(32)
                                                         ])

    vars_func = ll.Function(module, fnty_vars, name="vars")

    bb_entry_var = vars_func.append_basic_block()
    #bb_loop_var = func.append_basic_block()
    bb_exit_var = vars_func.append_basic_block()

    builder_var = ll.IRBuilder()
    builder_var.position_at_end(bb_entry_var)
    builder_var.branch(bb_exit_var)

    builder_var.position_at_end(bb_exit_var)
    # builder.ret_void()
    index0_var = builder_var.phi(ll.IntType(32))
    index0_var.add_incoming(ll.Constant(index0_var.type, 0), bb_entry_var)
    vg_ptr = builder_var.gep(var_global, [index0_var, index0_var])
    builder_var.ret(vg_ptr)


    strmod = str(module)

    t2 = time()

    print("-- generate IR:", t2-t1)

    t3 = time()
    #strmod)
    llmod = llvm.parse_assembly(strmod)

    t4 = time()

    print("-- parse assembly:", t4-t3)

    #print(llmod)

    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 9
    pm = llvm.create_module_pass_manager()
    pmb.populate(pm)

    t5 = time()

    pm.run(llmod)

    t6 = time()

    print("-- optimize:", t6-t5)

    t7 = time()




    #llvm.Target.triple="Linux "
    #target_machine = llvm.get_process_triple().create_target_machine()

    ee.add_module(llmod)

    ee.finalize_object()
    cfptr = ee.get_function_address("kernel")
    cfptr_var = ee.get_function_address("vars")

    t8 = time()
    print("-- JIT compile:", t8 - t7)

    #print(target_machine.emit_assembly(llmod))
    from numba import float32
    c_float_type =  c_float

    #diff_ = CFUNCTYPE(POINTER(c_float_type), POINTER(c_float_type))(cfptr)
    diff_ = CFUNCTYPE(POINTER(c_float_type), POINTER(c_float_type))(cfptr)

    vars_ = CFUNCTYPE(POINTER(c_float_type), c_int32)(cfptr_var)


    @njit('float32[:](float32[:])')
    def diff(y):

        deriv_pointer = diff_(y.ctypes)

        return carray(deriv_pointer, (max_deriv,))



    @njit('float32[:](float32[:],int32)')
    def variables_(var, set_):
        variables_pointer = vars_(0)
        variables_array = carray(variables_pointer, (max_var,))

        if set_>0:
            if len(var) == max_var:
               # print('setting vars: ', var[:])
                variables_array[:] = var[:]
            else:
                raise ValueError('bad length!')

        return variables_array


    y = np.ones(max_state, np.float32)

    diff(y)


    return diff, variables_, max_deriv

