cmake_minimum_required(VERSION 3.20.3)

project(EventAnalyze)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.6.2
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
  FindTBB
  GIT_REPOSITORY https://github.com/justusc/FindTBB.git
  GIT_TAG        25ecdea817b3af4a26d74ddcd439642dbd706acb
)
FetchContent_GetProperties(FindTBB)
if(NOT findtbb_POPULATED)
  FetchContent_Populate(FindTBB)
  list(APPEND CMAKE_MODULE_PATH "${findtbb_SOURCE_DIR}" ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
endif()

find_package(TBB)
if(TBB_FOUND)
  add_definitions(-DHAS_TBB)
  link_libraries(tbb)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  add_compile_options(-Wall -Wextra)
endif()

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "src/*.cpp")
pybind11_add_module(eventanalyze ${sources})
target_include_directories(eventanalyze PUBLIC include)
target_compile_features(eventanalyze PUBLIC cxx_std_20)
