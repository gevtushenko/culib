project(img_histogram CXX CUDA)

find_package(PNG)

if (PNG_FOUND)
    add_executable(${PROJECT_NAME})
    target_include_directories(${PROJECT_NAME} PRIVATE ${PNG_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${PNG_LIBRARY} culib)
    target_sources(${PROJECT_NAME} PRIVATE histogram.cu)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options -g,-march=native>)
endif()
