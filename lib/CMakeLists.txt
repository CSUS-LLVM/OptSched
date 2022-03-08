#IF(${PACKAGE_VERSION} VERSION_LESS "8.0")
#  add_llvm_loadable_module(OptSched OBJECT ${OPT_SCHED_SRCS})
#ELSE()
#  add_llvm_library(OptSched MODULE OBJECT ${OPT_SCHED_SRCS})
#ENDIF()

add_llvm_target(
  OptSched Scheduler/aco.cu
  Scheduler/bb_spill.cu
  Scheduler/buffers.cpp
  Scheduler/config.cpp
  Scheduler/data_dep.cu
  Scheduler/enumerator.cpp
  Scheduler/gen_sched.cu
  Scheduler/graph.cu
  Scheduler/graph_trans.cpp
  Scheduler/hist_table.cpp
  Scheduler/list_sched.cu
  Scheduler/logger.cpp
  Scheduler/reg_alloc.cpp
  Scheduler/utilities.cpp
  Scheduler/machine_model.cu
  Scheduler/random.cu
  Scheduler/ready_list.cu
  Scheduler/register.cu
  Scheduler/relaxed_sched.cpp
  Scheduler/sched_basic_data.cu
  Scheduler/sched_region.cu
  Scheduler/stats.cpp
  Wrapper/OptimizingScheduler.cu
  Wrapper/OptSchedMachineWrapper.cpp
  Wrapper/OptSchedDDGWrapperBasic.cpp
  Wrapper/OptSchedGenericTarget.cpp

  STATIC
)

#add_dependencies(OptSched ${OPT_SCHED_TARGET_DEPS})

target_compile_definitions(LLVMOptSched PRIVATE INSERT_ON_STEPFRWRD)
target_include_directories(LLVMOptSched PRIVATE ${OPTSCHED_TARGET_DEPS})