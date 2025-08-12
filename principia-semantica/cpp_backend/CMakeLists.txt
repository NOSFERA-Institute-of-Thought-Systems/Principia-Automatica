cmake_minimum_required(VERSION 3.12)
project(cpp_backend)

# This is the modern, recommended way to handle dependencies like pybind11.
# It fetches the library directly from GitHub at build time, ensuring
# a consistent and self-contained build without relying on the system's
# pybind11 installation.
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.10.1 # Pin to a specific version for stability
)
FetchContent_MakeAvailable(pybind11)

# Define the Python module.
# pybind11_add_module will automatically handle the necessary include directories
# and linker settings for pybind11.
#
# The first argument is the name of the Python module that will be generated.
# In Python, you will import it as 'cpp_backend'.
#
# Note: You no longer need to manually find Python3 or pybind11. The
# FetchContent approach handles finding the necessary dependencies for you.
pybind11_add_module(cpp_backend
  src/main.cpp
)

# Optionally, you can specify link libraries if your C++ code
# has other dependencies. For example:
# target_link_libraries(cpp_backend PRIVATE MyOtherLibrary)
