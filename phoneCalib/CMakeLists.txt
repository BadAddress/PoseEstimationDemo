cmake_minimum_required(VERSION 2.8)
project(vo1)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++17 -O2 ${SSE_FLAGS} -msse4")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 4 REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        "/usr/include/eigen3/"
)

add_executable(calib calib.cpp)
target_link_libraries(calib ${OpenCV_LIBS})
