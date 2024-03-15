cmake_minimum_required(VERSION 3.10)
project(BA_OPT)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_MODULE_PATH "/home/bl/3rdParty/g2o/cmake_modules/")



# 查找需要的包
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)
find_package(G2O REQUIRED)

# 包含头文件
include_directories(${CERES_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS})
include_directories("/usr/local/include/eigen3")


# 添加可执行文件
add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
# 链接库
target_link_libraries(pose_estimation_3d2d ${CERES_LIBRARIES} ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${Sophus_INCLUDE_DIRS} fmt::fmt g2o_core g2o_stuff)



add_executable(pose_estimation_3d3d pose_estimation_3d2d.cpp)
target_link_libraries(pose_estimation_3d3d ${CERES_LIBRARIES} ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${Sophus_INCLUDE_DIRS} fmt::fmt g2o_core g2o_stuff)


# 添加可执行文件
add_executable(direct_method direct_method.cpp)
# 链接库
target_link_libraries(direct_method ${CERES_LIBRARIES} ${Pangolin_LIBRARIES} ${OpenCV_LIBS} ${Sophus_INCLUDE_DIRS} fmt::fmt g2o_core g2o_stuff)
