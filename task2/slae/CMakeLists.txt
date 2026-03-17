cmake_minimum_required(VERSION 3.10)
project(slae_omp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

add_executable(slae_section slae_section.cpp)
target_link_libraries(slae_section PRIVATE OpenMP::OpenMP_CXX)

add_executable(slae_schedule slae_schedule.cpp)
target_link_libraries(slae_schedule PRIVATE OpenMP::OpenMP_CXX)
