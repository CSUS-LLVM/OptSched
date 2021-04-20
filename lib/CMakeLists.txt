if(${PACKAGE_VERSION} VERSION_LESS "8.0")
  add_llvm_loadable_module(OptSched OBJECT ${OPT_SCHED_SRCS})
else()
  add_llvm_library(OptSched MODULE OBJECT ${OPT_SCHED_SRCS})
endif()

add_dependencies(OptSched ${OPT_SCHED_TARGET_DEPS})

target_link_libraries(OptSched -L/home/paul/ROCm/opencl/build/lib/ libamdocl64.so)
