set(LOCAL_LLVM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/.LocalLLVM
  CACHE PATH "")
set(LOCAL_LLVM_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})
mark_as_advanced(LOCAL_LLVM_BINARY_DIR LOCAL_LLVM_LIST_DIR)

set(LOCAL_LLVM_DIR ${LOCAL_LLVM_BINARY_DIR}/llvm-project
  CACHE PATH "Path to a local copy of llvm-project, for use in unit tests and the AMDGPU build")

function(get_local_llvm)
  if(NOT EXISTS LOCAL_LLVM_BINARY_DIR)
    file(MAKE_DIRECTORY ${LOCAL_LLVM_BINARY_DIR})
  endif()

  if(NOT EXISTS LOCAL_LLVM_DIR AND
    NOT EXISTS ${LOCAL_LLVM_BINARY_DIR}/llvm-project.download-finished)
    if(NOT EXISTS ${LOCAL_LLVM_BINARY_DIR}/llvm-project.zip)
      file(DOWNLOAD https://github.com/CSUS-LLVM/llvm-project/archive/optsched.zip
        ${LOCAL_LLVM_BINARY_DIR}/llvm-project.zip
        SHOW_PROGRESS
        STATUS result
        TLS_VERIFY ON
        EXPECTED_HASH SHA256=c3a2e966d7182c031973530c0c8e010235577025ca54bfe8159d721f05ca2ed4
      )
      list(GET 0 result downloadFailed)
      list(GET 1 result statusString)

      if(downloadFailed)
        message(FATAL_ERROR "Unable to get llvm-project. Failed with ${downloadFailed}: ${statusString}")
      endif()
    endif()

    if(EXISTS ${LOCAL_LLVM_BINARY_DIR}/llvm-project-optsched)
      file(REMOVE_RECURSE ${LOCAL_LLVM_BINARY_DIR}/llvm-project-optsched)
    endif()

    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf llvm-project.zip
      WORKING_DIRECTORY ${LOCAL_LLVM_BINARY_DIR}
      RESULTS_VARIABLE unzipError
    )
    if(unzipError)
      message(FATAL_ERROR "Unable to unzip llvm-project. Failed with ${unzipError}")
    endif()

    file(RENAME ${LOCAL_LLVM_BINARY_DIR}/llvm-project-optsched ${LOCAL_LLVM_BINARY_DIR}/llvm-project)

    # Touch the file. file(TOUCH ...) is CMake 3.12+, but we want to support CMake 3.10
    file(WRITE ${LOCAL_LLVM_BINARY_DIR}/llvm-project.download-finished "")
  endif()

  cmake_parse_arguments(ARG "GTEST" "" "" ${ARGN})

  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown arguments ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  set(llvm_dir ${LOCAL_LLVM_DIR}/llvm)
  set(llvm_build_dirs ${LOCAL_LLVM_BINARY_DIR}/llvm_build_dirs)

  file(MAKE_DIRECTORY ${llvm_build_dirs})

  if(ARG_GTEST)
    # Set things up so that llvm-lit can do its work
    set(LLVM_EXTERNAL_LIT "${llvm_dir}/utils/lit/lit.py" CACHE PATH "Path to llvm-lit")
    add_subdirectory(${llvm_dir}/utils/unittest ${llvm_build_dirs}/googletest)

    # Set up GTest include dirs
    include_directories(
      ${llvm_dir}/utils/unittest/googletest/include
      ${llvm_dir}/utils/unittest/googlemock/include
    )
  endif()
endfunction()
