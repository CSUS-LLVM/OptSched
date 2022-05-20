set(OPTSCHED_SRCS
  Scheduler/aco.cpp
  Scheduler/bb_spill.cpp
  Scheduler/buffers.cpp
  Scheduler/config.cpp
  Scheduler/data_dep.cpp
  Scheduler/enumerator.cpp
  Scheduler/gen_sched.cpp
  Scheduler/graph.cpp
  Scheduler/graph_trans.cpp
  Scheduler/graph_trans_ilp.cpp
  Scheduler/graph_trans_ilp_occupancy_preserving.cpp
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
  Wrapper/OptSchedGenericTarget.cpp
)

set(OPTSCHED_TARGET_DEPS "")

if(OPTSCHED_ENABLE_AMDGPU)
  list(APPEND OPTSCHED_SRCS
    Wrapper/AMDGPU/GCNOptSched.cpp
    Wrapper/AMDGPU/OptSchedGCNTarget.cpp
    Wrapper/AMDGPU/OptSchedDDGWrapperGCN.cpp
  )
  if(TARGET LLVMAMDGPUCodeGen)
    list(APPEND OPTSCHED_TARGET_DEPS AMDGPUCommonTableGen)
  endif()
endif()

add_llvm_target(OptSched
  STATIC
  ${OPTSCHED_SRCS}
)
add_dependencies(LLVMOptSched ${OPTSCHED_TARGET_DEPS})
#add_definitions(${OPTSCHED_EXTRA_DEFINITIONS})
