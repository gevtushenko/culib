project(examples_common CXX CUDA)

find_package(PNG REQUIRED)

add_library(${PROJECT_NAME}
        include/png_reader.h
        src/png_reader.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        include

        PRIVATE
        ${PNG_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE ${PNG_LIBRARY})
