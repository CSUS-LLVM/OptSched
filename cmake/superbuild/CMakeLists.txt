# A SuperBuild project for OptSched.
#
# Use this as the CMake source directory to run the superbuild.
#
# Documentation of configuration args:
#
# In general, see option()s and set(... CACHE ...)s in this CMakeLists.txt and flang.cmake.
#
# Some options behave as follows:
#  - OPTSCHEDSUPER_FLANG: Check out and build flang.
#  - These options are passed on to llvm (and OptSched) directly from the SuperBuild configuration:
#      - CMAKE_BUILD_TYPE. Note that the flang build types default to Release, but can be configured.
#      - LLVM_PARALLEL_LINK_JOBS.
#  - *_EXTRA_CMAKE_ARGS: Passes these CMake arguments on to the corresponding sub-build.
#  - The flang builds can be configured to use a custom CMAKE_GENERATOR, separate from the superbuild's generator.
cmake_minimum_required(VERSION 3.7)

project(OptSched-SuperBuild)

option(OPTSCHEDSUPER_FLANG "Check out and build flang." OFF)

set(OPTSCHEDSUPER_LLVM_INSTALL_PREFIX   ${CMAKE_CURRENT_BINARY_DIR}/llvm-install      CACHE PATH   "The directory to install llvm")
set(OPTSCHEDSUPER_LLVM_GIT_REPO         https://github.com/CSUS-LLVM/llvm-project.git CACHE STRING "The git repository to clone for LLVM")
set(OPTSCHEDSUPER_LLVM_GIT_TAG          optsched                                      CACHE STRING "The tag to checkout within the LLVM git repository")
set(OPTSCHEDSUPER_LLVM_SOURCE_SUBDIR    llvm                                          CACHE STRING "LLVM lives in this subdirectory of the cloned repository")
set(OPTSCHEDSUPER_LLVM_EXTRA_CMAKE_ARGS ""                                            CACHE STRING ";-separated list of extra arguments to pass to configure cmake")

include(ExternalProject)
include(flang.cmake)
include(CTest)

# add_cache_variables(<OUT> <VARS>...)
# Compute a cache variable argument for each VARS, adding them to OUT.
function(add_cache_variables OUT_LIST)
  set(result "${${OUT_LIST}}")

  foreach(NAME IN LISTS ARGN)
    if(DEFINED ${NAME})
      get_property(type CACHE ${NAME} PROPERTY TYPE)
      if(NOT type)
        set(type UNINITIALIZED)
      endif()
      list(APPEND result -D${NAME}:${type}=${${NAME}})
    endif()
  endforeach()

  set(${OUT_LIST} "${result}" PARENT_SCOPE)
endfunction()

set(cache_default_args)
add_cache_variables(cache_default_args
  CMAKE_CXX_COMPILER_LAUNCHER
  CMAKE_C_COMPILER_LAUNCHER
  OPTSCHED_EXTRA_DEFINITIONS
  OPTSCHED_EXTRA_LINK_LIBRARIES
)

set(ROOT_DIR ${PROJECT_SOURCE_DIR}/../..)

ExternalProject_Add(llvm
  GIT_REPOSITORY ${OPTSCHEDSUPER_LLVM_GIT_REPO}
  GIT_TAG ${OPTSCHEDSUPER_LLVM_GIT_TAG}
  SOURCE_SUBDIR ${OPTSCHEDSUPER_LLVM_SOURCE_SUBDIR}
  INSTALL_DIR ${OPTSCHEDSUPER_LLVM_INSTALL_PREFIX}
  CMAKE_ARGS
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}
    -DLLVM_ENABLE_PROJECTS='clang'
    -DLLVM_TARGETS_TO_BUILD=X86
    -DLLVM_BUILD_TOOLS=ON
    -DLLVM_INCLUDE_TESTS=ON
    -DLLVM_OPTIMIZED_TABLEGEN=ON
    -DLLVM_EXTERNAL_PROJECTS=OptSched
    -DLLVM_EXTERNAL_OPTSCHED_SOURCE_DIR=${ROOT_DIR}
    ${OPTSCHEDSUPER_LLVM_EXTRA_CMAKE_ARGS}
  CMAKE_CACHE_DEFAULT_ARGS
    ${cache_default_args}
  USES_TERMINAL_CONFIGURE 1
  USES_TERMINAL_BUILD 1
  USES_TERMINAL_INSTALL 1
)

if(OPTSCHEDSUPER_FLANG)
  setup_flang_external_projects("${cache_default_args}")
endif()

ExternalProject_Get_Property(llvm BINARY_DIR)
set(LLVM_BUILD_DIR ${BINARY_DIR})

add_test(NAME OptSchedTests
  COMMAND ${CMAKE_COMMAND} --build . --target check-optsched
  WORKING_DIRECTORY ${LLVM_BUILD_DIR}
)
add_test(NAME OptSched-CompileHelloWorld
  COMMAND
    ${OPTSCHEDSUPER_LLVM_INSTALL_PREFIX}/bin/clang ${CMAKE_CURRENT_SOURCE_DIR}/helloworld.cpp
    -O3
    -fplugin=${OPTSCHEDSUPER_LLVM_INSTALL_PREFIX}/lib/OptSched.so
    -mllvm -misched=optsched
    -mllvm -enable-misched
    -mllvm -optsched-cfg=${ROOT_DIR}/example/optsched-cfg
)
