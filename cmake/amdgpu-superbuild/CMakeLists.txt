cmake_minimum_required(VERSION 3.7)

project(OptSched-AMDGPU-SuperBuild)

option(OPTSCHEDSUPER_AUTO_DOWNLOAD_REPO "Automatically download Google repo" OFF)
set(OPTSCHED_CFG $ENV{HOME}/optsched-cfg CACHE PATH "Path to the optsched-cfg directory you wish to use")

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  message(WARNING "Clang is required to build ROCm OpenCL. This build is likely to fail.")
endif()

if(OPTSCHEDSUPER_AUTO_DOWNLOAD_REPO)
  set(REPO_DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}/repo/download)
  set(REPO_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/repo/bin)

  if(NOT EXISTS ${REPO_DOWNLOAD_DIR}/repo)
    message(STATUS "Downloading repo...")
    file(DOWNLOAD https://storage.googleapis.com/git-repo-downloads/repo ${REPO_DOWNLOAD_DIR}/repo)
  endif()
  if(NOT EXISTS ${REPO_BIN_DIR}/repo)
    message(STATUS "Setting up repo...")
    file(COPY ${REPO_DOWNLOAD_DIR}/repo DESTINATION ${REPO_BIN_DIR}/
      PATTERN * PERMISSIONS OWNER_EXECUTE)
  endif()
endif()

find_program(REPO_COMMAND repo HINTS $ENV{HOME}/bin PATHS ${REPO_BIN_DIR})
if(NOT REPO_COMMAND)
  message(FATAL_ERROR
    "Google repo is required to run the AMDGPU superbuild.\n"
    "Either add a path containing repo in CMAKE_PROGRAM_PATH or CMAKE_PREFIX_PATH, "
    "or set -DOPTSCHEDSUPER_AUTO_DOWNLOAD_REPO=ON"
    )
endif()
message(STATUS "Found repo: ${REPO_COMMAND}")

find_program(SED_COMMAND sed)
if(NOT SED_COMMAND)
  message(FATAL_ERROR "sed is required to update the ROCm repo as needed for our build")
endif()
message(STATUS "Found sed: ${SED_COMMAND}")

include(ExternalProject)

function(add_variable OUT_LIST NAME)
  set(var "${${OUT_LIST}}")

  if(DEFINED ${NAME})
    get_property(type CACHE ${NAME} PROPERTY TYPE)
    if(NOT type)
      set(type UNINITIALIZED)
    endif()
    list(APPEND var -D${NAME}:${type}=${${NAME}})
  endif()

  set(${OUT_LIST} "${var}" PARENT_SCOPE)
endfunction()

function(add_variables OUT_LIST)
  set(list)

  foreach(var IN LISTS ARGN)
    add_variable(list ${var})
  endforeach()

  set(${OUT_LIST} "${list}" PARENT_SCOPE)
endfunction()

set(cache_default_args)
add_variables(cache_default_args
  CMAKE_CXX_COMPILER
  CMAKE_C_COMPILER
  CMAKE_CXX_COMPILER_LAUNCHER
  CMAKE_C_COMPILER_LAUNCHER
)

set(ROOT_DIR ${PROJECT_SOURCE_DIR}/../..)
set(PATCH_DIR ${ROOT_DIR}/patches/ROCm)

ExternalProject_Add(ROCm
  DOWNLOAD_COMMAND  cd <SOURCE_DIR>
                    && ${REPO_COMMAND} init -u https://github.com/RadeonOpenCompute/ROCm-OpenCL-Runtime.git
                      -b roc-2.4.x -m opencl.xml
  PATCH_COMMAND ${REPO_COMMAND} sync
        COMMAND # Remove the line "local:".
                ${SED_COMMAND} -i "/local:/d" <SOURCE_DIR>/opencl/api/opencl/amdocl/amdocl.map
        COMMAND cd <SOURCE_DIR>/opencl/compiler/driver && git apply ${PATCH_DIR}/load-plugins.patch
  SOURCE_SUBDIR opencl
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DLLVM_PARALLEL_LINK_JOBS=${LLVM_PARALLEL_LINK_JOBS}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    "'-DLLVM_TARGETS_TO_BUILD=X86;AMDGPU'"
    -DLLVM_BUILD_TOOLS=ON
    -DLLVM_INCLUDE_TESTS=ON
    -DCLANG_ENABLE_STATIC_ANALYZER=ON
    "-DOPTSCHED_EXTRA_LINK_LIBRARIES=<BINARY_DIR>/lib/libamdocl64.so"
  CMAKE_CACHE_DEFAULT_ARGS
    ${cache_default_args}
  USES_TERMINAL_CONFIGURE 1
  USES_TERMINAL_BUILD 1
  USES_TERMINAL_INSTALL 1
  # Don't install anything
  INSTALL_COMMAND ""
)
# Add symlink to OptSched inside llvm/projects
ExternalProject_Add_Step(ROCm optsched_symlink
  DEPENDEES patch
  DEPENDERS configure
  COMMAND
    ${CMAKE_COMMAND} -E create_symlink
      ${ROOT_DIR} <SOURCE_DIR>/opencl/compiler/llvm/projects/OptSched
  BYPRODUCTS <SOURCE_DIR>/opencl/compiler/llvm/projects/OptSched
  USES_TERMINAL 1
)
ExternalProject_Get_Property(ROCm BINARY_DIR)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/rocm-env.shrc"
  "export LD_LIBRARY_PATH=${BINARY_DIR}/lib\n"
  "export OPTLIB=${BINARY_DIR}/compiler/llvm/lib/OptSched.so\n"
  "export AMD_OCL_BUILD_OPTIONS_APPEND=\"-Wf,-fplugin=$OPTLIB -Wb,-misched=gcn-optsched -Wb,-optsched-cfg=${OPTSCHED_CFG}\"\n"
)

ExternalProject_Add_Step(ROCm inform_env_exports
  DEPENDEES install
  COMMAND ${CMAKE_COMMAND} -E echo
    "Don't forget to edit your .bashrc to `source ${CMAKE_CURRENT_BINARY_DIR}/rocm-env.shrc`"
  BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/rocm-env.shrc
  USES_TERMINAL 1
)
