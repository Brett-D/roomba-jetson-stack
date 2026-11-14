cmake_minimum_required(VERSION 3.16)
project(basalt_vio_ros CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(isaac_ros_visual_slam_interfaces REQUIRED)
find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc features2d)

# Basalt (Monado fork) source tree built with `cmake --preset release`.
set(BASALT_ROOT "$ENV{HOME}/src/basalt" CACHE PATH "Basalt source tree")
set(BASALT_LIB "${BASALT_ROOT}/build/libbasalt_internal.a")
if(NOT EXISTS "${BASALT_LIB}")
  message(FATAL_ERROR "Build Basalt first: ${BASALT_LIB} not found")
endif()

add_executable(basalt_vio_node src/basalt_vio_node.cpp)
target_compile_features(basalt_vio_node PRIVATE cxx_std_17)
# Must match how libbasalt_internal.a was compiled (Eigen alignment/vectorization and the
# template instantiations it contains), see build/build.ninja for basalt_vio.
target_compile_definitions(basalt_vio_node PRIVATE
  BASALT_INSTANTIATIONS_DOUBLE BASALT_INSTANTIATIONS_FLOAT FMT_DEPRECATED_OSTREAM=1 EIGEN_DONT_PARALLELIZE)
target_compile_options(basalt_vio_node PRIVATE -march=native -O3 -Wno-deprecated-declarations)
# Basalt's bundled Eigen/Sophus first, before any system copy pulled in by other packages.
target_include_directories(basalt_vio_node SYSTEM BEFORE PRIVATE
  ${BASALT_ROOT}/include
  ${BASALT_ROOT}/thirdparty/basalt-headers/include
  ${BASALT_ROOT}/thirdparty/basalt-headers/thirdparty/eigen
  ${BASALT_ROOT}/thirdparty/basalt-headers/thirdparty/Sophus
  ${BASALT_ROOT}/thirdparty/basalt-headers/thirdparty/cereal/include
  ${BASALT_ROOT}/thirdparty/magic_enum/include
  ${BASALT_ROOT}/thirdparty/vit
  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(basalt_vio_node
  ${BASALT_LIB} ${BASALT_ROOT}/build/lib/libopengv.a TBB::tbb fmt::fmt ${OpenCV_LIBS} lz4 stdc++fs
  rclcpp::rclcpp ${sensor_msgs_TARGETS} ${nav_msgs_TARGETS} ${geometry_msgs_TARGETS}
  tf2_ros::tf2_ros ${isaac_ros_visual_slam_interfaces_TARGETS})
target_link_options(basalt_vio_node PRIVATE -fuse-ld=mold)

install(TARGETS basalt_vio_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

ament_package()
