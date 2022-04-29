add_llvm_target(
  OptSched Scheduler/aco.hip.cpp
  Scheduler/simplified_aco_ds.hip.cpp
  Scheduler/bb_spill.hip.cpp
  Scheduler/buffers.cpp
  Scheduler/config.cpp
  Scheduler/data_dep.hip.cpp
  Scheduler/enumerator.cpp
  Scheduler/gen_sched.hip.cpp
  Scheduler/graph.hip.cpp
  Scheduler/graph_trans.cpp
  Scheduler/hist_table.cpp
  Scheduler/list_sched.hip.cpp
  Scheduler/logger.cpp
  Scheduler/reg_alloc.cpp
  Scheduler/utilities.cpp
  Scheduler/machine_model.hip.cpp
  Scheduler/random.hip.cpp
  Scheduler/ready_list.hip.cpp
  Scheduler/register.hip.cpp
  Scheduler/relaxed_sched.cpp
  Scheduler/sched_basic_data.hip.cpp
  Scheduler/sched_region.hip.cpp
  Scheduler/stats.cpp
  Wrapper/OptimizingScheduler.hip.cpp
  Wrapper/OptSchedMachineWrapper.cpp
  Wrapper/OptSchedDDGWrapperBasic.cpp
  Wrapper/OptSchedGenericTarget.cpp
  Wrapper/AMDGPU/GCNOptSched.cpp
  Wrapper/AMDGPU/OptSchedGCNTarget.cpp
  Wrapper/AMDGPU/OptSchedDDGWrapperGCN.cpp
  STATIC
)
target_include_directories(LLVMOptSched
    PRIVATE
        ${HIP_PATH}/include
        ${HIP_PATH}/../include
        ${OPTSCHED_TARGET_DEPS})

target_link_options(LLVMOptSched PUBLIC "$<$<LINK_LANGUAGE:CXX>:-fgpu-rdc;--hip-link>")
target_link_libraries(LLVMOptSched ${HIP_LIBRARIES})

set(OPTSCHED_TARGET_DEPS ${OPTSCHED_SOURCE_DIR}/include ${OPTSCHED_SOURCE_DIR}/lib)

target_compile_definitions(LLVMOptSched PRIVATE INSERT_ON_STEPFRWRD)
target_include_directories(LLVMOptSched PRIVATE ${OPTSCHED_TARGET_DEPS})
