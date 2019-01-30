IF (${PACKAGE_VERSION} VERSION_LESS "8.0")
add_llvm_loadable_module(OptSched
  ${OPT_SCHED_SRCS}
  )
ELSE ()
add_llvm_library(OptSched MODULE
  ${OPT_SCHED_SRCS}
  )
ENDIF()

add_dependencies(OptSched ${OPT_SCHED_TARGET_DEPS})
