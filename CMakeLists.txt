cmake_minimum_required(VERSION 3.15)
project(culib)

set(CMAKE_CXX_STANDARD 17)

# CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Library
add_library(${PROJECT_NAME} INTERFACE)
target_sources(${PROJECT_NAME}
        INTERFACE
            include/culib/warp/scan.cuh
            include/culib/device/scan.cuh
            include/culib/warp/reduce.cuh
            include/culib/warp/detail/warp_reduce.cuh
            include/culib/warp/detail/warp_shuffle.cuh
            include/culib/warp/utils.cuh
            include/culib/warp/shuffle.cuh
            include/culib/warp/compact.cuh
            include/culib/block/reduce.cuh
            include/culib/utils/meta/any.cuh
            include/culib/utils/meta/math.cuh
            include/culib/warp/detail/warp_scan.cuh)

target_include_directories(${PROJECT_NAME}
        INTERFACE
            include)

target_compile_options(${PROJECT_NAME}
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# Add the cmake folder so the FindSphinx module is found
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Google Test
enable_testing()
add_subdirectory(test)
add_subdirectory(docs)
