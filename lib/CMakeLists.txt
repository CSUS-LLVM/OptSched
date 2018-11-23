add_llvm_loadable_module(OptSched
  ${OPT_SCHED_SRCS}
  PLUGIN_TOOL
  llc
  )

add_dependencies(OptSched intrinsics_gen)
