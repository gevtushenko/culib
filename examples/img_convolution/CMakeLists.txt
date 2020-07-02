project(img_convolution CXX CUDA)

add_executable(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE examples_common culib)
target_sources(${PROJECT_NAME} PRIVATE convolution.cu)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -g --compiler-options -g,-mavx2>)
