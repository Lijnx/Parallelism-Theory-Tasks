cmake_minimum_required(VERSION 3.16)
project(task6_heat2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(heat2d heat2d.cpp)
target_link_libraries(heat2d PRIVATE Boost::program_options)

option(ENABLE_OPENACC "Compile with OpenACC flags" ON)
set(OPENACC_TARGET "gpu" CACHE STRING "NVIDIA HPC SDK OpenACC target: gpu, multicore or host")
set_property(CACHE OPENACC_TARGET PROPERTY STRINGS gpu multicore host)

if(ENABLE_OPENACC)
    target_compile_options(heat2d PRIVATE -acc=${OPENACC_TARGET} -Minfo=accel)
    target_link_options(heat2d PRIVATE -acc=${OPENACC_TARGET})
endif()

target_compile_options(heat2d PRIVATE -O2)

message(STATUS "OpenACC enabled: ${ENABLE_OPENACC}")
message(STATUS "OpenACC target:  ${OPENACC_TARGET}")
