project(img_minmax CXX CUDA)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(NCCL)

add_executable(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE examples_common culib ${NCCL_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${NCCL_INCLUDE_DIRS})
target_sources(${PROJECT_NAME} PRIVATE minmax.cu)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -g,-march=native>)
