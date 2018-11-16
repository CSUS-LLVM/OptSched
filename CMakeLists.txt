cmake_minimum_required(VERSION 3.4.3)

# Exit if attempting to build as a standalone project.
IF (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  message(FATAL_ERROR "You cannot build opt-sched as a standalone project. Place opt-sched in the llvm tree under the \"projects\" directory.")
ENDIF()

IF (NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "Must set cmake build type.")
ENDIF ()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add OptSched debug defines eg: SET(OPT_SCHED_DEBUG_FLAGS "-DIS_DEBUG_DEFS_AND_USES -DIS_DEBUG_DEF_USE_COUNT")
SET(OPT_SCHED_DEBUG_FLAGS "")

# If asserts are enabled opt-sched must be built with "IS_DEBUG".
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DIS_DEBUG")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_SCHED_DEBUG_FLAGS}")

IF (CMAKE_BUILD_TYPE EQUAL "DEBUG")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
ENDIF ()

add_subdirectory(lib)
