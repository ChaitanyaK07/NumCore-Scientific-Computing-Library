cmake_minimum_required(VERSION 3.15)

project(NumCore VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Use modern Python discovery with pybind11
set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Python extension module
pybind11_add_module(_NumCore
    python/bindings.cpp
)

target_include_directories(_NumCore PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

install(TARGETS _NumCore
    LIBRARY DESTINATION numcore
    RUNTIME DESTINATION numcore
    ARCHIVE DESTINATION numcore
)



# Compiler warnings for Python module
if (MSVC)
    target_compile_options(_NumCore PRIVATE /W4)
else()
    target_compile_options(_NumCore PRIVATE -Wall -Wextra)
endif()

# Pure C++ unit-test binary
add_executable(tests_cpp
    tests/tests_cpp.cpp
)

target_include_directories(tests_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Compiler warnings for C++ tests
if (MSVC)
    target_compile_options(tests_cpp PRIVATE /W4)
else()
    target_compile_options(tests_cpp PRIVATE -Wall -Wextra)
endif()