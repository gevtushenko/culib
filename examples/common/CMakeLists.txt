project(examples_common CXX CUDA)

find_package(PNG)

add_library(${PROJECT_NAME}
        include/png_reader.h
        src/png_reader.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC include)

if(PNG_FOUND)
    target_compile_definitions(${PROJECT_NAME} PUBLIC PNG_PRESENT=1)
    target_include_directories(${PROJECT_NAME} PRIVATE ${PNG_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${PNG_LIBRARY})
endif()
