add_library(OptSched.Lib OBJECT ${OPT_SCHED_SRCS})

IF (${PACKAGE_VERSION} VERSION_LESS "8.0")
  add_llvm_loadable_module(OptSched $<TARGET_OBJECTS:OptSched.Lib>)
ELSE ()
  add_llvm_library(OptSched MODULE $<TARGET_OBJECTS:OptSched.Lib>)
ENDIF()

add_dependencies(OptSched.Lib ${OPT_SCHED_TARGET_DEPS})
