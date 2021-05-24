cmake_minimum_required(VERSION 3.4.3)

project(OptSched)

option(OPTSCHED_INCLUDE_TESTS "Generate build targets for the OptSched unit tests." ON)
option(OPTSCHED_ENABLE_AMDGPU "Build the AMDGPU code. Requires that the AMDGPU target is supported." OFF)
set(OPTSCHED_LIT_ARGS "-sv" CACHE STRING "Arguments to pass to lit")
set(OPTSCHED_EXTRA_LINK_LIBRARIES "" CACHE STRING "Extra link_libraries to pass to OptSched, ;-separated")
set(OPTSCHED_EXTRA_INCLUDE_DIRS "" CACHE STRING "Extra include_directories to pass to OptSched, ;-separated")
# To add OptSched debug defines, e.g.:
# '-DOPTSCHED_EXTRA_DEFINITIONS=-DIS_DEBUG_DEFS_AND_USES;-DIS_DEBUG_DEF_USE_COUNT'
set(OPTSCHED_EXTRA_DEFINITIONS "" CACHE STRING "Extra add_definitions to pass to OptSched, ;-separated")

if(TARGET LLVMCodeGen)
  set(llvm_subproject TRUE)
else()
  set(llvm_subproject FALSE)
endif()

if(NOT llvm_subproject)
  set(llvm_version 6.0)
  if(OPTSCHED_ENABLE_AMDGPU)
    set(llvm_version 9.0)
  endif()

  set(OPTSCHED_LLVM_VERSION ${llvm_version} CACHE STRING "The LLVM version to build OptSched with (independent build only)")

  find_package(LLVM ${OPTSCHED_LLVM_VERSION} REQUIRED CONFIG)
endif()

if(NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "Must set cmake build type.")
endif()

# If we are requested to build the AMDGPU functionality, ensure that we can do so.
if(OPTSCHED_ENABLE_AMDGPU)
  if(NOT "AMDGPU" IN_LIST LLVM_ALL_TARGETS)
    message(FATAL_ERROR "Trying to build the AMDGPU code, but AMDGPU is not supported by this build of LLVM")
  elseif(LLVM_VERSION VERSION_LESS 7.0)
    message(FATAL_ERROR "OptSched requries LLVM version >= 7.0 to build the AMDGPU scheduler.")
  endif()
endif()

# If asserts are enabled opt-sched must be built with "IS_DEBUG".
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DIS_DEBUG")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(OPTSCHED_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OPTSCHED_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(AddLLVM)

include_directories(
  ${OPTSCHED_SOURCE_DIR}/include
  ${OPTSCHED_SOURCE_DIR}/lib
  ${LLVM_INCLUDE_DIRS}
  ${OPTSCHED_EXTRA_INCLUDE_DIRS}
)
add_definitions(${OPTSCHED_EXTRA_DEFINITIONS})
link_directories(${OPTSCHED_EXTRA_LINK_LIBRARIES})

if(LLVM_VERSION VERSION_LESS 7.0)
  add_definitions(-DLLVM_DEBUG=DEBUG)
endif()

if(NOT llvm_subproject)
  include(GetLocalLLVM)

  set(args)
  if(OPTSCHED_INCLUDE_TESTS)
    list(APPEND args GTEST)
  endif()

  get_local_llvm(${args})
endif()

add_subdirectory(lib)

if(OPTSCHED_INCLUDE_TESTS)
  include(CTest)

  add_subdirectory(unittests)
  list(APPEND OPTSCHED_TEST_DEPS OptSchedUnitTests)
  list(APPEND OPTSCHED_TEST_PARAMS
    clang_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/test/Unit/lit.site.cfg
    )

  add_subdirectory(test)

  add_test(NAME OptSchedTests
    COMMAND ${CMAKE_COMMAND} --build . --target check-optsched
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_test(NAME OptSched-CompileHelloWorld
    COMMAND
      ${LLVM_TOOLS_BINARY_DIR}/clang ${CMAKE_CURRENT_SOURCE_DIR}/example/helloworld.cpp
      -O3
      -fplugin=$<TARGET_FILE:OptSched>
      -mllvm -misched=optsched
      -mllvm -enable-misched
      -mllvm -optsched-cfg=${CMAKE_CURRENT_SOURCE_DIR}/example/optsched-cfg
  )
endif()
