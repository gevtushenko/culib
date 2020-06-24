cmake_minimum_required(VERSION 3.15)
project(culib)

set(CMAKE_CXX_STANDARD 11)

# culib version
find_package(Git QUIET)
if (GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} describe
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE "GIT_VERSION"
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
    string(REGEX REPLACE "-g.*" "" GIT_VERSION "${GIT_VERSION}")
    set(CULIB_VERSION ${GIT_VERSION})
endif()

configure_file("${PROJECT_SOURCE_DIR}/cmake/version.cuh.in" "${CMAKE_SOURCE_DIR}/include/culib/utils/version.cuh" @ONLY)

# CUDA
find_package(CUDA REQUIRED)

# Library
add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
    PUBLIC
        src/node/communication.cpp

        include/culib/warp/scan.cuh
        include/culib/device/memory/resizable_array.h
        include/culib/device/memory/pointer.h
        include/culib/device/memory/api.h
        include/culib/device/scan.cuh
        include/culib/device/histogram.cuh
        include/culib/device/reduce.cuh
        include/culib/node/communication.h
        include/culib/node/reduce.cuh
        include/culib/warp/reduce.cuh
        include/culib/warp/detail/warp_reduce.cuh
        include/culib/warp/detail/warp_shuffle.cuh
        include/culib/warp/utils.cuh
        include/culib/warp/shuffle.cuh
        include/culib/warp/compact.cuh
        include/culib/block/histogram.cuh
        include/culib/block/reduce.cuh
        include/culib/utils/meta/any.cuh
        include/culib/utils/meta/binary_ops.cuh
        include/culib/utils/meta/math.cuh
        include/culib/utils/meta/limits.cuh
        include/culib/warp/detail/warp_scan.cuh)

target_include_directories(${PROJECT_NAME}
        PUBLIC
            "${CUDA_INCLUDE_DIRS}"
            include)

if(CUDA_VERSION_MAJOR LESS 7)
target_compile_options(${PROJECT_NAME} INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_11,code=sm_11>)
endif()

# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

option(BUILD_TESTS "Build tests for CULIB" ON)

# Google Test
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
    add_subdirectory(docs)
endif()

option(BUILD_BENCHMARKS "Build benchmarks for CULIB" OFF)

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

add_subdirectory(examples)
