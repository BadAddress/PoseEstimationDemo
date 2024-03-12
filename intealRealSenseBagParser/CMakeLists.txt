cmake_minimum_required(VERSION 3.1)
project(RealsenseBagReader)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)

# 寻找OpenCV库
find_package(OpenCV REQUIRED)

# 包含RealSense库
find_package(realsense2 REQUIRED)

# 包含目录
include_directories(${OpenCV_INCLUDE_DIRS})

# 定义可执行文件
add_executable(parser parser.cpp)

# 链接库
target_link_libraries(parser ${realsense2_LIBRARY} ${OpenCV_LIBS})
