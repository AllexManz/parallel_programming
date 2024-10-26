cmake_minimum_required(VERSION 3.28)
project(Lab2 C)

set(CMAKE_C_COMPILER /opt/homebrew/Cellar/gcc@11/11.5.0/bin/gcc-11)
set(CMAKE_C_STANDARD 11)

find_package(OpenMP REQUIRED)

add_executable(Lab2 lab2.c)

# Enable OpenMP support
if (OpenMP_C_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    target_link_libraries(Lab2 PRIVATE ${OpenMP_C_LIBRARIES})
endif ()