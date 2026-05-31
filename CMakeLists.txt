cmake_minimum_required(VERSION 3.18)
project(bzip2gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDA  REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

file(GLOB_RECURSE ALL_SRC "src/*.cpp" "src/*.cu" "src/*.c")

set(APP_ENTRY_POINT "src/main.cpp")

list(FILTER ALL_SRC EXCLUDE REGEX "${APP_ENTRY_POINT}")

add_library(bzip2gpu_core STATIC ${ALL_SRC})

target_include_directories(bzip2gpu_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_executable(bzip2gpu ${APP_ENTRY_POINT})
target_link_libraries(bzip2gpu PRIVATE bzip2gpu_core)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

file(GLOB_RECURSE TEST_SRC "tests/*.cpp" "tests/*.cu")

add_executable(unit_tests ${TEST_SRC})

target_link_libraries(unit_tests PRIVATE
    bzip2gpu_core
    gtest_main
)

add_test(NAME AllUnitTests COMMAND unit_tests)
