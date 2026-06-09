cmake_minimum_required(VERSION 3.16)
project(depth_normal_render LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# OpenCV >= 4.x.
find_package(OpenCV 4 REQUIRED)

# Eigen — header-only, look in standard system paths first, then
# CMAKE_PREFIX_PATH. Override with -DEigen3_DIR=... if needed.
find_package(Eigen3 3.3 NO_MODULE)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 REQUIRED)
endif()

add_executable(depth_normal_render
    Demo.cpp
    LUT.cpp
    Phong.cpp
)

target_include_directories(depth_normal_render PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(depth_normal_render PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

# std::filesystem on older GCC needs explicit linkage.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
    target_link_libraries(depth_normal_render PRIVATE stdc++fs)
endif()

if(MSVC)
    target_compile_options(depth_normal_render PRIVATE /W3 /permissive-)
else()
    target_compile_options(depth_normal_render PRIVATE -Wall -Wextra -Wpedantic)
endif()
