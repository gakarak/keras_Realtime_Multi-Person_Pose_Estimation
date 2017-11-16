cmake_minimum_required(VERSION 3.5)
project(read_hdr)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp def_hmp_data.hpp def_hmp_data.cpp)
add_executable(read_hdr ${SOURCE_FILES})