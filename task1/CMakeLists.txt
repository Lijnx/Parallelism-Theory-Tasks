cmake_minimum_required(VERSION 3.15)
project(SinSum VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_DOUBLE ON)
if (USE_DOUBLE)
    add_compile_definitions(USE_DOUBLE)
endif()

add_executable(exec_trg main.cpp)
