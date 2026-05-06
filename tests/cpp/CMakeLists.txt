cmake_minimum_required(VERSION 3.16)
project(remote_vlm_cpp_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
  FetchContent_MakeAvailable(googletest)
endif()

set(CLIENT_SRC ${CMAKE_SOURCE_DIR}/../../client/src)

# Build the units under test as a tiny static lib so individual test
# binaries can link without depending on the full client (which pulls
# X11, libdatachannel, gRPC, etc).
add_library(vlm_units STATIC
  ${CLIENT_SRC}/aff/adaptive_frame_filter.cpp
  ${CLIENT_SRC}/scd/scene_change_detector.cpp)
target_include_directories(vlm_units PUBLIC
  ${CLIENT_SRC}
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vlm_units PUBLIC ${OpenCV_LIBS} Threads::Threads)

enable_testing()

function(add_vlm_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE vlm_units)
  if(TARGET GTest::gtest_main)
    target_link_libraries(${name} PRIVATE GTest::gtest_main)
  else()
    target_link_libraries(${name} PRIVATE gtest_main)
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

add_vlm_test(test_adaptive_frame_filter test_adaptive_frame_filter.cpp)
add_vlm_test(test_scene_change_detector test_scene_change_detector.cpp)
add_vlm_test(test_capture_mode test_capture_mode.cpp)
