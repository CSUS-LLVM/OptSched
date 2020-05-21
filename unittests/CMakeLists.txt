add_custom_target(OptSchedUnitTests)
set_target_properties(OptSchedUnitTests PROPERTIES FOLDER "Tests")

# LLVM turns this off, but we're good to have it:
add_definitions(-UGTEST_HAS_TR1_TUPLE)

function(add_optsched_unittest test_dirname)
  add_unittest(OptSchedUnitTests ${test_dirname} ${ARGN})
endfunction()

# If built standalone, we need to get GTest.
if(NOT HAS_PARENT)
  include(FetchContent)

  if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/llvm_googletest.download-finished
    OR NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/llvm-project-optsched)
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/llvm_googletest.zip)
      file(DOWNLOAD https://github.com/CSUS-LLVM/llvm-project/archive/optsched.zip
        ${CMAKE_CURRENT_BINARY_DIR}/llvm_googletest.zip
        SHOW_PROGRESS
        STATUS result
        TLS_VERIFY ON
        EXPECTED_HASH SHA256=c3a2e966d7182c031973530c0c8e010235577025ca54bfe8159d721f05ca2ed4
      )
      list(GET 0 result downloadFailed)
      list(GET 1 result statusString)

      if(downloadFailed)
        message(FATAL_ERROR "Unable to get googletest. Failed with ${downloadFailed}: ${statusString}")
      endif()
    endif()
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/llvm_googletest)
      file(REMOVE_RECURSE ${CMAKE_CURRENT_BINARY_DIR}/llvm_googletest)
    endif()

    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzf llvm_googletest.zip
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULTS_VARIABLE unzipError
    )
    if(unzipError)
      message(FATAL_ERROR "Unable to unzip googletest. Failed with ${unzipError}")
    endif()

    file(TOUCH ${CMAKE_CURRENT_BINARY_DIR}/llvm_googletest.download-finished)
  endif()

  set(llvm_dir ${CMAKE_CURRENT_BINARY_DIR}/llvm-project-optsched/llvm)
  set(llvm_build_dirs ${CMAKE_CURRENT_BINARY_DIR}/llvm_build_dirs)

  # Set things up so that llvm-lit can do its work
  file(MAKE_DIRECTORY ${llvm_build_dirs})
  set(LLVM_EXTERNAL_LIT "${llvm_dir}/utils/lit/lit.py" CACHE PATH "Path to llvm-lit")

  add_subdirectory(${llvm_dir}/utils/unittest ${llvm_build_dirs}/googletest)

  # Set up GTest include dirs
  include_directories(
    ${llvm_dir}/utils/unittest/googletest/include
    ${llvm_dir}/utils/unittest/googlemock/include
  )
endif()

# All unit test targets depend on OptSched
add_library(UnitTest.OptSched STATIC $<TARGET_OBJECTS:obj.OptSched>)
link_libraries(UnitTest.OptSched)

add_subdirectory(Basic)
