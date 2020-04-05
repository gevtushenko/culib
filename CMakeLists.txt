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
            include/culib/device/scan.cuh)

target_include_directories(${PROJECT_NAME}
        INTERFACE
            include)

target_compile_options(${PROJECT_NAME}
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

# Google Test
enable_testing()
add_subdirectory(test)
