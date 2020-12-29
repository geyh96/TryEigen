cmake_minimum_required(VERSION 3.17)
project(TryEigen)

set(CMAKE_CXX_STANDARD 11)
include_directories(/usr/include/eigen3/)
set(SOURCE_FILES main.cpp)
add_executable(TryEigen main.cpp)