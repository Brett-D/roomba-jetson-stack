#!/usr/bin/env bash
# Regenerates src/{msg,srv}.rs and src/{msg,srv}/rmw.rs from the installed Isaac ROS interface
# definitions: builds a throwaway copy of the package (rosidl_generate_interfaces with the Rust
# generator, ros-jazzy-rosidl-generator-rs) in $WORK and copies only the generated Rust sources.
# The throwaway package is never installed or sourced; the bindings link NVIDIA's libraries.
set -e
source /opt/ros/jazzy/setup.bash
SHARE=/opt/ros/jazzy/share/isaac_ros_visual_slam_interfaces
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$HOME/.cache/isaac_rs_gen}
PKG=$WORK/src/isaac_ros_visual_slam_interfaces
rm -rf "$WORK"; mkdir -p "$PKG"
cp -r "$SHARE/msg" "$SHARE/srv" "$PKG/"
VERSION=$(sed -n 's:.*<version>\(.*\)</version>.*:\1:p' "$SHARE/package.xml")
cat > "$PKG/package.xml" <<EOF
<?xml version="1.0"?>
<package format="3">
  <name>isaac_ros_visual_slam_interfaces</name>
  <version>$VERSION</version>
  <description>Throwaway copy, only to generate Rust bindings</description>
  <maintainer email="none@example.com">none</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <build_depend>rosidl_default_generators</build_depend>
  <depend>std_msgs</depend>
  <depend>geometry_msgs</depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
  <export><build_type>ament_cmake</build_type></export>
</package>
EOF
cat > "$PKG/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.16)
project(isaac_ros_visual_slam_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
file(GLOB MSGS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} msg/*.msg)
file(GLOB SRVS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} srv/*.srv)
rosidl_generate_interfaces(${PROJECT_NAME} ${MSGS} ${SRVS} DEPENDENCIES std_msgs geometry_msgs)
ament_package()
EOF
(cd "$WORK" && colcon build --event-handlers console_cohesion- > build.log 2>&1) || { tail -20 "$WORK/build.log"; exit 1; }
GEN=$WORK/install/isaac_ros_visual_slam_interfaces/share/isaac_ros_visual_slam_interfaces/rust/src
mkdir -p "$HERE/src/msg" "$HERE/src/srv"
for f in msg.rs srv.rs msg/rmw.rs srv/rmw.rs; do cp "$GEN/$f" "$HERE/src/$f"; done
sed -i "s/^version = .*/version = \"$VERSION\"/" "$HERE/Cargo.toml"
rm -rf "$WORK"
echo "Regenerated from $SHARE ($VERSION)"