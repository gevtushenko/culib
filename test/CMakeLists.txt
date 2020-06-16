project(culib_test CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(gtest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
        PRIVATE
            src/gtest_runner.cpp
            include/test_helper.cuh
            src/block/reduce_test.cu
            src/device/reduce_test.cu
            src/device/pointer_test.cu
            src/warp/compact_test.cu
            src/warp/reduce_test.cu
            src/warp/scan_test.cu)
target_include_directories(${PROJECT_NAME}
        PRIVATE
            include)
target_link_libraries(${PROJECT_NAME}
        PRIVATE
            gtest
            ${CMAKE_PROJECT_NAME})

add_test(
        NAME core
        COMMAND culib_test)
