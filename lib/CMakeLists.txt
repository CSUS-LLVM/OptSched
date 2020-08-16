IF(${PACKAGE_VERSION} VERSION_LESS "8.0")
  add_llvm_loadable_module(OptSched OBJECT ${OPT_SCHED_SRCS})
ELSE()
  add_llvm_library(OptSched MODULE OBJECT ${OPT_SCHED_SRCS})
ENDIF()

add_dependencies(OptSched ${OPT_SCHED_TARGET_DEPS})
target_link_libraries(OptSched -L/home/vang/src/ROCm-2.4/opencl/build/lib/ libamdocl64.so)
