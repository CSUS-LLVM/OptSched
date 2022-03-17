#IF(${PACKAGE_VERSION} VERSION_LESS "8.0")
#  add_llvm_loadable_module(OptSched OBJECT ${OPT_SCHED_SRCS})
#ELSE()
#  add_llvm_library(OptSched MODULE OBJECT ${OPT_SCHED_SRCS})
#ENDIF()

#enable_language(CUDA)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Allow our .cpp files to include CUDA headers:
#include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

list(APPEND CMAKE_PREFIX_PATH /opt/rocm-4.5.2/hip /opt/rocm-4.5.2)

find_package(hip)
set(HIP_PATH "/opt/rocm-4.5.2/hip")

add_llvm_target(
  OptSched Scheduler/aco.hip.cpp
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
        ${HIP_PATH}/include>
        ${HIP_PATH}/../include>
        ${OPTSCHED_TARGET_DEPS})



target_link_libraries(LLVMOptSched hip::device)

set(OPTSCHED_TARGET_DEPS ${OPTSCHED_SOURCE_DIR}/include ${OPTSCHED_SOURCE_DIR}/lib "/opt/rocm-4.5.2/hiprand/include" "/opt/rocm-4.5.2/rocrand/include")
#add_dependencies(OptSched ${OPT_SCHED_TARGET_DEPS})

target_compile_definitions(LLVMOptSched PRIVATE INSERT_ON_STEPFRWRD)
target_include_directories(LLVMOptSched PRIVATE ${OPTSCHED_TARGET_DEPS})

#target_include_directories(LLVMOptSched PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
