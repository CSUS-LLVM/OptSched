add_custom_target(OptSchedUnitTests)
set_target_properties(OptSchedUnitTests PROPERTIES FOLDER "Tests")

# LLVM turns this off, but we're good to have it:
add_definitions(-UGTEST_HAS_TR1_TUPLE)

function(add_optsched_unittest test_dirname)
  add_unittest(OptSchedUnitTests ${test_dirname} ${ARGN})
endfunction()

# All unit test targets depend on OptSched
add_library(UnitTest.OptSched STATIC $<TARGET_OBJECTS:obj.OptSched>)
link_libraries(UnitTest.OptSched)

add_subdirectory(Basic)
