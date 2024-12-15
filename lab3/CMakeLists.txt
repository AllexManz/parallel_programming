cmake_minimum_required(VERSION 3.28)
project(Lab3 C)

# Set the C compiler
set(CMAKE_C_COMPILER /opt/homebrew/Cellar/gcc@11/11.5.0/bin/gcc-11)
set(CMAKE_C_STANDARD 11)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add the executable
add_executable(Lab3 lab3.c)

# Enable OpenMP support
if (OpenMP_C_FOUND)
    target_compile_options(Lab3 PRIVATE ${OpenMP_C_FLAGS})
    target_link_libraries(Lab3 PRIVATE OpenMP::OpenMP_C)
    # Explicitly set the linker flags for OpenMP
    set_target_properties(Lab3 PROPERTIES LINK_FLAGS "${OpenMP_C_FLAGS}")
endif ()