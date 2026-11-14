cmake_minimum_required(VERSION 3.16)
project(mynteye_d1000_driver)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)

# MYNT-EYE-D-SDK install prefix (built by mynt-eye-ros2/scripts/build_d_sdk_ubuntu24.sh)
set(MYNTEYE_D_SDK_ROOT "$ENV{HOME}/mynt-eye-ros2/d_install" CACHE PATH "MYNT-EYE-D-SDK install prefix")
list(APPEND CMAKE_PREFIX_PATH "${MYNTEYE_D_SDK_ROOT}")
find_package(mynteyed REQUIRED CONFIG)
set(ESPDI_LIB_DIR "${MYNTEYE_D_SDK_ROOT}/lib/3rdparty")

add_executable(d1000_node src/d1000_node.cpp)
target_compile_definitions(d1000_node PRIVATE WITH_OPENCV)
target_include_directories(d1000_node PRIVATE ${MYNTEYE_D_SDK_ROOT}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(d1000_node
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  tf2_ros::tf2_ros
  mynteye_depth
  ${OpenCV_LIBS})
set_target_properties(d1000_node PROPERTIES
  BUILD_RPATH "${MYNTEYE_D_SDK_ROOT}/lib;${ESPDI_LIB_DIR}"
  INSTALL_RPATH "${MYNTEYE_D_SDK_ROOT}/lib;${ESPDI_LIB_DIR}")

install(TARGETS d1000_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
