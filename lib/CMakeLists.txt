add_llvm_loadable_module(OptSched
  Scheduler/aco.cpp
  Scheduler/bb_spill.cpp
  Scheduler/buffers.cpp
  Scheduler/config.cpp
  Scheduler/data_dep.cpp
  Scheduler/enumerator.cpp
  Scheduler/gen_sched.cpp
  Scheduler/graph.cpp
  Scheduler/graph_trans.cpp
  Scheduler/hist_table.cpp
  Scheduler/list_sched.cpp
  Scheduler/logger.cpp
  Scheduler/reg_alloc.cpp
  Scheduler/utilities.cpp
  Scheduler/machine_model.cpp
  Scheduler/random.cpp
  Scheduler/ready_list.cpp
  Scheduler/register.cpp
  Scheduler/relaxed_sched.cpp
  Scheduler/sched_basic_data.cpp
  Scheduler/sched_region.cpp
  Scheduler/stats.cpp
  Wrapper/OptimizingScheduler.cpp
  Wrapper/OptSchedMachineWrapper.cpp
  Wrapper/OptSchedDDGWrapperBasic.cpp
  Wrapper/OptSchedDDGWrapperGCN.cpp
  Wrapper/OptSchedGenericTarget.cpp

  PLUGIN_TOOL
  llc
  )

add_dependencies(OptSched intrinsics_gen)
