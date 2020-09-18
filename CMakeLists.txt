cmake_minimum_required(VERSION 3.4.3)

option(OPTSCHED_INCLUDE_TESTS "Generate build targets for the OptSched unit tests." ON)
set(OPTSCHED_EXTRA_LINK_LIBRARIES "" CACHE STRING "Extra link_libraries to pass to OptSched, ;-separated")
# To add OptSched debug defines, e.g.:
# '-DOPTSCHED_EXTRA_DEFINITIONS=-DIS_DEBUG_DEFS_AND_USES;-DIS_DEBUG_DEF_USE_COUNT'
set(OPTSCHED_EXTRA_DEFINITIONS "" CACHE STRING "Extra add_definitions to pass to OptSched, ;-separated")

# Exit if attempting to build as a standalone project.
get_directory_property(hasParent PARENT_DIRECTORY)
if(NOT hasParent)
  message(FATAL_ERROR "You cannot build opt-sched as a standalone project. Place opt-sched in the llvm tree under the \"projects\" directory.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "Must set cmake build type.")
endif()

if(${PACKAGE_VERSION} VERSION_LESS "7.0")
  add_definitions("-DLLVM_DEBUG=DEBUG")
endif()

# If asserts are enabled opt-sched must be built with "IS_DEBUG".
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DIS_DEBUG")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OPTSCHED_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OPTSCHED_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# OptSched sources
set(OPT_SCHED_SRCS Scheduler/aco.cpp
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
  Wrapper/OptSchedGenericTarget.cpp)

set(OPT_SCHED_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(OPT_SCHED_TARGET_DEPS "LLVMCodeGen")

function(check_if_AMDGPU_supported)
  if(${PACKAGE_VERSION} VERSION_LESS "7.0")
    message(WARNING "OptSched requries LLVM version >= 7.0 to build the AMDGPU scheduler.")
    set(AMDGPU_SUPPORTED False PARENT_SCOPE)

  else()
    set(AMDGPU_SUPPORTED True PARENT_SCOPE)

  endif()
endfunction(check_if_AMDGPU_supported)

# Only build and include AMDGPU functionality if it is being targeted.
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  check_if_AMDGPU_supported()
  if(${AMDGPU_SUPPORTED})
    set(OPT_SCHED_TARGET_DEPS ${OPT_SCHED_TARGET_DEPS} AMDGPUCommonTableGen)
    set(OPT_SCHED_SRCS ${OPT_SCHED_SRCS} Wrapper/AMDGPU/GCNOptSched.cpp
                                          Wrapper/AMDGPU/OptSchedGCNTarget.cpp
                                          Wrapper/AMDGPU/OptSchedDDGWrapperGCN.cpp)
    set(OPT_SCHED_INCLUDE_DIRS ${OPT_SCHED_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/lib
                                ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
                                ${LLVM_BINARY_DIR}/lib/Target/AMDGPU)
  endif()
endif()

include_directories(${OPT_SCHED_INCLUDE_DIRS})
add_definitions(${OPTSCHED_EXTRA_DEFINITIONS})
link_libraries(${OPTSCHED_EXTRA_LINK_LIBRARIES})
add_subdirectory(lib)

if(OPTSCHED_INCLUDE_TESTS)
  add_subdirectory(unittests)
  list(APPEND OPTSCHED_TEST_DEPS OptSchedUnitTests)
  list(APPEND OPTSCHED_TEST_PARAMS
    clang_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/test/Unit/lit.site.cfg
    )

  add_subdirectory(test)
endif()
