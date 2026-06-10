cmake_minimum_required(VERSION 3.15)

project(NumCore VERSION 0.1.3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(PYBIND11_FINDPYTHON ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED)

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

if(MSVC)
    target_compile_options(_NumCore PRIVATE /W4)
else()
    target_compile_options(_NumCore PRIVATE -Wall -Wextra)
endif()

option(NUMCORE_BUILD_TESTS "Build NumCore C++ tests" OFF)

if(NUMCORE_BUILD_TESTS)
    add_executable(tests_cpp
        tests/tests_cpp.cpp
    )

    target_include_directories(tests_cpp PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )

    if(MSVC)
        target_compile_options(tests_cpp PRIVATE /W4)
    else()
        target_compile_options(tests_cpp PRIVATE -Wall -Wextra)
    endif()
endif()