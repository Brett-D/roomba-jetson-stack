#!/usr/bin/env bash
# Stop the SLAM / robot launch and every node it started.
pkill -INT -f "ros2 bag record" 2>/dev/null
pkill -INT -f "ros2 launch roomba_bringup" 2>/dev/null
# rtabmap saves its database on shutdown (can take a while for big maps): wait for it
# before anything gets killed harder, or the map can be truncated.
for i in $(seq 1 90); do pgrep -f "lib/rtabmap_slam/rtabmap" > /dev/null || break; sleep 1; done
pkill -f "/opt/ros/jazzy/lib/nav2_" 2>/dev/null
sleep 3
pkill -f 'run_handheld_slam.sh|run_vslam.sh|ros2 launch roomba_bringup' 2>/dev/null
pkill -f '/opt/ros/jazzy/lib/(rtabmap_|foxglove_bridge|imu_filter_madgwick|image_transport|rclcpp_components)' 2>/dev/null
pkill -f 'mynteye_d1000_driver/d1000_node' 2>/dev/null
pkill -f "roomba_perception/" 2>/dev/null
pkill -f "basalt_vio_ros/" 2>/dev/null
pkill -f "roomba_web/" 2>/dev/null
pkill -INT -f "lib/roomba_driver/roomba_driver" 2>/dev/null
pkill -f "tf2_ros/static_transform_publisher" 2>/dev/null
sleep 5
pkill -9 -f "/opt/ros/jazzy/lib/(rtabmap_|foxglove_bridge|imu_filter_madgwick|image_transport|rclcpp_components)" 2>/dev/null
pkill -9 -f "mynteye_d1000_driver/d1000_node" 2>/dev/null
pkill -9 -f "roomba_(perception|web|driver)/|basalt_vio_ros/|static_transform_publisher" 2>/dev/null
pkill -9 -f "/opt/ros/jazzy/lib/nav2_" 2>/dev/null
sleep 1
pgrep -af 'rtabmap|foxglove|d1000_node|imu_filter|republish|component_container|person_detector|ignore_zones|vslam_watchdog|basalt_vio_node|web_server|nav2_|floor_obstacles|nav_coach|explorer|roomba_driver|static_transform' | grep -v pgrep || echo "all stopped"
