cmake_minimum_required(VERSION 3.16)
project(roomba_rs)

# The nodes are a plain cargo project (rclrs + ros-env, which finds the message crates on
# AMENT_PREFIX_PATH). CMake only drives cargo so colcon, ros2 run and launch files work as usual.
# (colcon-ros-cargo's ament_cargo build type breaks here: it patches in crates for Isaac ROS
# message packages that ship no Rust bindings.)
find_package(ament_cmake REQUIRED)
find_program(CARGO cargo HINTS "$ENV{HOME}/.cargo/bin" REQUIRED)

set(CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo")
set(NODES floor_obstacles ekf_inputs vslam_watchdog map_gap_filler ignore_zones web_server set_password explorer nav_coach person_detector twin2d nav2_guard sound_events battery_learner)

# Learned navigation skills (docs/learned-skills.md): on by default since the robot uses them
# (robot.launch.py: unstuck_skill:=true goal_ranking:=learned). ON adds the cargo feature "learned"
# (tract ONNX inference, the explorer's goal_ranking: learned), the unstuck_skill node and the models.
# CMake caches the choice: an existing build directory configured OFF needs -DLEARNED_SKILLS=ON once.
option(LEARNED_SKILLS "Build the learned navigation skills" ON)
set(CARGO_FEATURES "")
if(LEARNED_SKILLS)
  list(APPEND NODES unstuck_skill)
  set(CARGO_FEATURES --features learned)
  install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)
endif()

add_custom_target(cargo_build ALL
  COMMAND ${CMAKE_COMMAND} -E env --unset=MAKEFLAGS --unset=MFLAGS CARGO_TARGET_DIR=${CARGO_TARGET_DIR} ${CARGO} build --release --locked ${CARGO_FEATURES}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  USES_TERMINAL)

foreach(node ${NODES})
  install(PROGRAMS ${CARGO_TARGET_DIR}/release/${node} DESTINATION lib/${PROJECT_NAME})
endforeach()

if(BUILD_TESTING)
  add_test(NAME cargo_test
    COMMAND ${CMAKE_COMMAND} -E env CARGO_TARGET_DIR=${CARGO_TARGET_DIR} ${CARGO} test --release --locked --lib ${CARGO_FEATURES}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

ament_package()