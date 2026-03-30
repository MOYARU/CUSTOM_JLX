cmake_minimum_required(VERSION 3.15)
project(CustomJLX LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Metal and Foundation frameworks
find_library(METAL_LIBRARY Metal)
find_library(FOUNDATION_LIBRARY Foundation)

include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.mm")
# Exclude backup/original main files
list(FILTER SOURCES EXCLUDE REGEX "main_.*\\.mm$")

add_executable(custom_jlx ${SOURCES})

# Link against Metal and Foundation
target_link_libraries(custom_jlx PRIVATE ${METAL_LIBRARY} ${FOUNDATION_LIBRARY})

# copy kernels to build directory
file(GLOB KERNELS "kernels/*.metal")
file(COPY ${KERNELS} DESTINATION ${CMAKE_BINARY_DIR}/kernels)
