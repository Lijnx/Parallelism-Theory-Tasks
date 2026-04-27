cmake_minimum_required(VERSION 3.10)

project(client_server_tbb LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TBB REQUIRED)

add_executable(client_server_app
    src/main.cpp
)

target_include_directories(client_server_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(client_server_app PRIVATE
    TBB::tbb
)

include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(client_server_tests
    tests/test_results.cpp
)

target_include_directories(client_server_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(client_server_tests PRIVATE
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(client_server_tests)