cmake_minimum_required(VERSION 3.11)
project(culib_benchmarks CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# CUDA
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Library
add_executable(${PROJECT_NAME} main.cu)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
target_link_libraries(${PROJECT_NAME} ${CMAKE_PROJECT_NAME})
