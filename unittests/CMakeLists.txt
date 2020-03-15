add_custom_target(OptSchedUnitTests)
set_target_properties(OptSchedUnitTests PROPERTIES FOLDER "Tests")

function(add_optsched_unittest test_dirname)
  add_unittest(OptSchedUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Basic)
