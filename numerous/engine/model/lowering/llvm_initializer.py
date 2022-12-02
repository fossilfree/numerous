import faulthandler
import llvmlite.binding as llvm

faulthandler.enable()
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()
llvmmodule = llvm.parse_assembly("")
target_machine = llvm.Target.from_default_triple().create_target_machine()
ee = llvm.create_mcjit_compiler(llvmmodule, target_machine)