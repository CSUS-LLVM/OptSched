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
  Scheduler/simplified_aco_ds.cpp
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

function(add_optsched_library name)
  if(LLVM_VERSION VERSION_LESS 8.0)
    add_llvm_loadable_module(${name} OBJECT ${ARGN})
  else()
    add_llvm_library(${name} MODULE OBJECT ${ARGN})
  endif()
endfunction()

add_optsched_library(OptSched
  ${OPTSCHED_SRCS}
  LINK_LIBS ${OPTSCHED_LINK_LIBS}
)
add_dependencies(OptSched ${OPTSCHED_TARGET_DEPS})
