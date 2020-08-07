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
def generate(program, functions, variables, variable_values, ix_d, n_deriv):


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
        #print('func: ', f['signature'])
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


    max_deriv = np.int32(n_deriv)
    max_state = np.int32(n_deriv)
    max_var = np.int32(len(variables))



    var_global = ll.GlobalVariable(module, ll.ArrayType(ll.FloatType(), max_var), 'global_var')
    var_global.initializer = ll.Constant(ll.ArrayType(ll.FloatType(), max_var), [float(v) for v in variable_values])




    func = ll.Function(module, fnty, name="kernel")

    bb_entry = func.append_basic_block()
    bb_loop = func.append_basic_block()
    bb_exit = func.append_basic_block()

    builder = ll.IRBuilder()
    builder.position_at_end(bb_entry)

    builder.branch(bb_loop)
    builder.position_at_end(bb_loop)
    #izero = builder.phi(ll.IntType(32))
    #izero.add_incoming(ll.Constant(izero.type, 0), bb_entry)

    #var_global = ll.GlobalVariable(module, ll.ArrayType(ll.FloatType(), max_deriv), 'var_global')
    #var_global.initializer = ll.Constant(ll.ArrayType(ll.FloatType(), max_deriv), [0] * max_deriv)
    #var_deriv = ll.GlobalVariable(module, ll.ArrayType(ll.FloatType(), max_deriv), 'var_deriv')
    #var_deriv.initializer = ll.Constant(ll.ArrayType(ll.FloatType(), max_deriv), [0] * max_deriv)
    #max_ret = 1000
    #var_returns = ll.GlobalVariable(module, ll.ArrayType(ll.FloatType(), max_ret), 'global_returns')
    #var_returns.initializer= ll.Constant(ll.ArrayType(ll.FloatType(), max_ret), [0] * max_ret)

    #var_deriv = ll.GlobalVariable(module, ll.ArrayType(ll.FloatType(), max_deriv), 'var_deriv')
    #var_deriv.initializer = ll.Constant(ll.ArrayType(ll.FloatType(), max_deriv), [0] * max_deriv)



    #index0 = builder.phi(ll.IntType(32))
    #index0.add_incoming(ll.Constant(index0.type, 0), bb_entry)

    #zero = builder.phi(ll.FloatType())
    #zero.add_incoming(ll.Constant(zero.type, 0), bb_entry)

    #m1 = builder.phi(ll.FloatType())
    #m1.add_incoming(ll.Constant(m1.type, -1), bb_entry)

    #p1 = builder.phi(ll.FloatType())
    #p1.add_incoming(ll.Constant(p1.type, 1), bb_entry)

    index0 = builder.phi(ll.IntType(32))
    index0.add_incoming(ll.Constant(index0.type, 0), bb_entry)
    #variables_pointer = builder.gep(variables, [index0])


    #for i, p in enumerate(program):
    #    print(i, ': ', p)
    values = {}
    for ix, p in enumerate(program):#+program[14:]:
        checks = []
        if 'args' in p:
            checks += p['args']
        if 'targets' in p:
            checks += p['targets']

        for i, t in enumerate(checks):
            if not t in values:
                ix = variables.index(t)
                index = builder.phi(ll.IntType(32), name=t + f"_ix_{ix =}")
                index.add_incoming(ll.Constant(index.type, ix), bb_entry)

                ptr = var_global
                ix = [index0, index]
                eptr = builder.gep(ptr, ix, name=t)
                values[t] = eptr

        if 'args' in p:
            #print(p['args'])
            args = [builder.load(values[a], 'arg_' + a) for a in p['args']]


        if p['func'] == 'load' or p['func'] == 'store':
            index = builder.phi(ll.IntType(32), name=p['arg'] + f"_ix_{p['ix']}")
            index.add_incoming(ll.Constant(index.type, p['ix']), bb_entry)

            if p['arg'] == 'variables':
                ptr = var_global
                ix = [index0, index]

            elif p['arg'] == 'y':
                ptr = func.args[0]
                ix = [index]


            if p['func'] == 'load':
                eptr = builder.gep(ptr, ix, name=p['arg']+"_"+p['var'])
                values[p['var']]=eptr

                if p['arg'] == 'y':
                    ptrg = var_global
                    index = builder.phi(ll.IntType(32))
                    index.add_incoming(ll.Constant(index.type, variables.index(p['var'])), bb_entry)
                    ix = [index0, index]
                    geptr = builder.gep(ptrg, ix)

                    builder.store(builder.load(values[p['var']]), geptr)

            elif p['func'] == 'save':

                builder.store(values[p['var']], eptr)





        elif p['func'] == 'call':
            #arr = builder.phi(ll.ArrayType(ll.FloatType(), 1).as_pointer())

            #print(p['ext_func'])


                    #raise ValueError('AAARG')
            #if not ix in [13, 15]:



            builder.call(ext_funcs[p['ext_func']], args + [values[t] for t in p['targets']])
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

                    builder.store(args[0], values[t])
                    #values[t] = values[p['args'][0]]
            elif la == 2:
                sum_ = builder.fadd(args[0], args[1])
                for t in p['targets']:


                    builder.store(sum_, values[t])
                    #values[t] = values[t]
            else:
                accum = builder.fadd(args[0], args[1])
                for i, a in enumerate(args[2:]):
                    accum = builder.fadd(accum, a)




                for t in p['targets']:

                    builder.store(accum, values[t])

            #sdfsdf=sdfsdf

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
    #builder.ret(builder.gep(var_deriv, [index0, index0]))

    indexd = builder.phi(ll.IntType(32))
    #print(variables)
    #print('len vars: ', max_var)
    #print('derivative ix: ', ix_d)
    #print(variables[ix_d:ix_d+n_deriv])
    indexd.add_incoming(ll.Constant(indexd.type, ix_d), bb_entry)
    dg_ptr = builder.gep(var_global, [index0, indexd])

    builder.ret(dg_ptr)
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
    #print(strmod)
    llmod = llvm.parse_assembly(strmod)

    t4 = time()

    print("-- parse assembly:", t4-t3)

    #print(llmod)

    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 1000
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

        return carray(deriv_pointer, (n_deriv,))



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

    @njit('float32[:](float32)')
    def variables__(f):
        variables_pointer = vars_(0)
        variables_array = carray(variables_pointer, (max_var,))

        return variables_array




    return diff, variables__, variables_, max_deriv

