cmake_minimum_required(VERSION 3.4.3)

# Exit if attempting to build as a standalone project.
if( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  message( FATAL_ERROR "You cannot build opt-sched as a standalone project. Place opt-sched in the llvm tree under the \"projects\" directory." )
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add OptSched debug defines
SET(OPT_SCHED_DEBUG_FLAGS "-DIS_DEBUG")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPT_SCHED_DEBUG_FLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

add_subdirectory(lib)
