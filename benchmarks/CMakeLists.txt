cmake_minimum_required(VERSION 3.11)
project(culib_benchmarks CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Library
add_executable(${PROJECT_NAME} src/main.cu include/results.h src/results.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -gencode arch=compute_75,code=sm_75>)
target_link_libraries(${PROJECT_NAME} ${CMAKE_PROJECT_NAME} stdc++fs)
target_include_directories(${PROJECT_NAME} PRIVATE include)
