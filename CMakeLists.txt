cmake_minimum_required(VERSION 3.18)

# project(OptSched LANGUAGES CXX CUDA)
#enable_language(CUDA)

# Allow our .cpp files to include CUDA headers:
#include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

include(CTest)

option(OPTSCHED_INCLUDE_TESTS "Generate build targets for the OptSched unit tests." ON)

# Exit if attempting to build as a standalone project.
IF(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "You cannot build opt-sched as a standalone project. Place opt-sched in the llvm tree under the \"projects\" directory.")
ENDIF()

IF(NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "Must set cmake build type.")
ENDIF()

IF(${PACKAGE_VERSION} VERSION_LESS "7.0")
  add_definitions("-DLLVM_DEBUG=DEBUG")
ENDIF()

# Add OptSched debug defines eg: SET(OPT_SCHED_DEBUG_FLAGS "-DIS_DEBUG_DEFS_AND_USES -DIS_DEBUG_DEF_USE_COUNT")
SET(OPT_SCHED_DEBUG_FLAGS "")

# If asserts are enabled opt-sched must be built with "IS_DEBUG".
SET(CMAKE_CXX_FLAGS_DEBUG "")

#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

IF(CMAKE_BUILD_TYPE EQUAL "DEBUG")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
ENDIF()

set(OPTSCHED_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OPTSCHED_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# OptSched sources
SET(OPT_SCHED_SRCS
  Scheduler/aco.hip.cpp
  Scheduler/bb_spill.hip.cpp
  Scheduler/data_dep.hip.cpp
  Scheduler/gen_sched.hip.cpp
  Scheduler/graph.hip.cpp
  Scheduler/list_sched.hip.cpp
  Scheduler/machine_model.hip.cpp
  Scheduler/random.hip.cpp
  Scheduler/ready_list.hip.cpp
  Scheduler/register.hip.cpp
  Scheduler/sched_basic_data.hip.cpp
  Scheduler/sched_region.hip.cpp
  Scheduler/simplified_aco_ds.hip.cpp
  Wrapper/OptimizingScheduler.hip.cpp

  Scheduler/buffers.cpp
  Scheduler/config.cpp
  Scheduler/enumerator.cpp
  Scheduler/graph_trans.cpp
  Scheduler/hist_table.cpp
  Scheduler/logger.cpp
  Scheduler/reg_alloc.cpp
  Scheduler/utilities.cpp
  Scheduler/relaxed_sched.cpp
  Scheduler/stats.cpp
  Wrapper/OptSchedMachineWrapper.cpp
  Wrapper/OptSchedDDGWrapperBasic.cpp
  Wrapper/OptSchedGenericTarget.cpp)


SET(OPT_SCHED_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/lib)
SET(OPT_SCHED_TARGET_DEPS "LLVMCodeGen")
set(llvm_subproject TRUE)

function(check_if_AMDGPU_supported)
  IF(${PACKAGE_VERSION} VERSION_LESS "7.0")
    message(WARNING "OptSched requries LLVM version >= 7.0 to build the AMDGPU scheduler.")
    SET(AMDGPU_SUPPORTED False PARENT_SCOPE)

  ELSE()
    SET(AMDGPU_SUPPORTED True PARENT_SCOPE)

  ENDIF()
endfunction(check_if_AMDGPU_supported)

# Only build and include AMDGPU functionality if it is being targeted.
IF("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  check_if_AMDGPU_supported()
  IF(${AMDGPU_SUPPORTED})
    SET(OPT_SCHED_TARGET_DEPS ${OPT_SCHED_TARGET_DEPS} AMDGPUCommonTableGen)
    SET(OPT_SCHED_SRCS ${OPT_SCHED_SRCS} Wrapper/AMDGPU/GCNOptSched.cpp
                                          Wrapper/AMDGPU/OptSchedGCNTarget.cpp
                                          Wrapper/AMDGPU/OptSchedDDGWrapperGCN.cpp)
    SET(OPT_SCHED_INCLUDE_DIRS ${OPT_SCHED_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/lib
                                ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
                                ${CMAKE_CURRENT_BINARY_DIR}/../../lib/Target/AMDGPU
                                ${CMAKE_CURRENT_BINARY_DIR}/..)
  ENDIF()
ENDIF()
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include_directories(${OPT_SCHED_INCLUDE_DIRS})
add_definitions(${OPTSCHED_EXTRA_DEFINITIONS})
link_directories(${OPTSCHED_EXTRA_LINK_LIBRARIES})

add_subdirectory(lib)
