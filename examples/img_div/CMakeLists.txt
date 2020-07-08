project(img_div CXX CUDA)

add_executable(${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} PRIVATE examples_common culib)
target_sources(${PROJECT_NAME} PRIVATE div.cu)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -dlcm=ca --compiler-options -O3,-march=native>)
