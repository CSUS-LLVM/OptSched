set(OPTSCHED_SRCS Scheduler/aco.cpp
  Scheduler/bb_spill.cpp
  Scheduler/buffers.cpp
  Scheduler/config.cpp
  Scheduler/data_dep.cpp
  Scheduler/enumerator.cpp
  Scheduler/gen_sched.cpp
  Scheduler/graph.cpp
  Scheduler/graph_trans.cpp
  Scheduler/graph_trans_ilp.cpp
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

set(OPTSCHED_TARGET_DEPS LLVMCodeGen)

if(OPTSCHED_ENABLE_AMDGPU)
  list(APPEND OPTSCHED_SRCS
    Wrapper/AMDGPU/GCNOptSched.cpp
    Wrapper/AMDGPU/OptSchedGCNTarget.cpp
    Wrapper/AMDGPU/OptSchedDDGWrapperGCN.cpp
  )
  if(TARGET AMDGPUCommonTableGen)
    list(APPEND OPTSCHED_TARGET_DEPS AMDGPUCommonTableGen)
  endif()
  if(llvm_subproject)
    include_directories(
      ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
      ${LLVM_BINARY_DIR}/lib/Target/AMDGPU
    )
  endif()
endif()

add_dependencies(OptSched ${OPT_SCHED_TARGET_DEPS})
