# Native build on the target machine:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
#   ctest --test-dir build --output-on-failure
cmake_minimum_required(VERSION 3.13)
project(newch C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(NC_NATIVE "Tune for the build machine's CPU (-mcpu/-march=native)" OFF)
option(NC_BUILD_SHARED "Also build the shared library used by the Python package" ON)
option(NC_BUILD_TESTS "Build the test suite" ON)
option(NC_BUILD_EXAMPLES "Build examples" ON)

find_package(Threads REQUIRED)
file(GLOB_RECURSE NC_SOURCES CONFIGURE_DEPENDS engine/src/*.c)

function(nc_configure target)
  target_include_directories(${target} PUBLIC engine/include)
  target_link_libraries(${target} PUBLIC Threads::Threads)
  if(NOT WIN32)
    target_link_libraries(${target} PUBLIC m)
  endif()
  if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    if(NC_NATIVE)
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
        target_compile_options(${target} PRIVATE -mcpu=native)
      else()
        target_compile_options(${target} PRIVATE -march=native)
      endif()
    endif()
  endif()
endfunction()

add_library(nc STATIC ${NC_SOURCES})
nc_configure(nc)

if(NC_BUILD_SHARED)
  # Only NC_API functions are exported; copy the result into newch/ for Python.
  add_library(nc_shared SHARED ${NC_SOURCES})
  nc_configure(nc_shared)
  set_target_properties(nc_shared PROPERTIES OUTPUT_NAME nc C_VISIBILITY_PRESET hidden)
  target_compile_definitions(nc_shared PRIVATE NC_BUILD_SHARED)
endif()

if(NC_BUILD_TESTS)
  enable_testing()
  add_executable(nc_tests engine/tests/test_main.c)
  target_link_libraries(nc_tests PRIVATE nc)
  add_test(NAME nc_tests COMMAND nc_tests)
endif()

if(NC_BUILD_EXAMPLES)
  add_executable(train_spiral examples/train_spiral.c)
  target_link_libraries(train_spiral PRIVATE nc)
endif()
