# CMakeLists.txt for building MPI benchmarking code
cmake_minimum_required(VERSION 3.10)
project(mpi_sum_compare LANGUAGES CXX)

# Set C++ standard and optimization flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS " -O3")

# Use MPI compiler
find_package(MPI REQUIRED)

# Set the compiler to MPI's C++ compiler
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MPI_CXX_INCLUDE_PATH}
)

configure_file(run_mpi_sum_compare.sl.in run_mpi_sum_compare.sl)

# Define the executable
add_executable(mpi_sum_compare mpi_sum_compare.cpp)

# Link MPI libraries
target_link_libraries(mpi_sum_compare )

