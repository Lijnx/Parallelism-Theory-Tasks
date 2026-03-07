cmake_minimum_required(VERSION 3.10)
project(dgemv_omp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

add_executable(dgemv dgemv.cpp)
target_link_libraries(dgemv PRIVATE OpenMP::OpenMP_CXX)
