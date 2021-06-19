cmake_minimum_required(VERSION 3.20.3)

project(EventAnalyze)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
  mio
  GIT_REPOSITORY https://github.com/mandreyel/mio.git
  GIT_TAG        3f86a95c0784d73ce6815237ec33ed25f233b643
)
FetchContent_MakeAvailable(mio)

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

pybind11_add_module(eventanalyze src/module.cpp src/parse.cpp src/types.cpp)
target_include_directories(eventanalyze PUBLIC include)
target_link_libraries(eventanalyze PRIVATE mio::mio)
target_compile_features(eventanalyze PUBLIC cxx_std_20)

add_executable(test src/test.main.cpp)
target_link_libraries(test PRIVATE mio::mio)
target_compile_features(test PRIVATE cxx_std_20)
