cmake_minimum_required(VERSION 3.20)
project(tool1_roofline_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tool1_roofline src/main.cpp)
