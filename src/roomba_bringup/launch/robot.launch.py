"""The Roomba robot: camera + VSLAM + Roomba base driver + Nav2, all in one launch.

  ros2 launch roomba_bringup robot.launch.py mode:=mapping         # drive around to build a new map
  ros2 launch roomba_bringup robot.launch.py                       # reuse the saved map (auto)
  ros2 launch roomba_bringup robot.launch.py nav:=false            # VSLAM + driver only

Frames: map (RTAB-Map) -> odom (vslam_watchdog: VIO, wheel-bridged) -> base_link (Roomba centre
on the floor, x forward) -> camera_link (static mount below) -> optical / IMU frames (camera driver).
roomba_driver drives the wheels from /cmd_vel (Nav2's collision monitor, or the web app's drive pad)
and publishes /wheel/odom, /battery_state and /roomba/sensors.

Camera mount defaults: centred, facing forward, level, lens 0.153 m ahead of the Roomba's centre
(1.7 cm behind the front bumper edge) and 0.13 m above the floor (tape measure). In the depth
data the floor is ~0.11 m below the lens (depth runs ~15% short); floor_obstacles uses that
calibrated floor_height directly because the mount is rigid.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# Passed through to vslam.launch.py unchanged.
VSLAM_ARGS = ['mode', 'odometry', 'database_path', 'foxglove', 'people', 'web', 'web_port',
              'ir_depth_only', 'ir_intensity', 'ground_constraint']


def generate_launch_description():
    bringup = FindPackageShare('roomba_bringup')
    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument('mode', default_value='auto',
                              description='auto (reuse saved map if present), mapping (new map), localization'),
        DeclareLaunchArgument('odometry', default_value='basalt'),
        DeclareLaunchArgument('ground_constraint', default_value='true',
                              description='cuVSLAM only: keep the odometry in the floor plane'),
        DeclareLaunchArgument('database_path', default_value=os.path.expanduser('~/maps/vslam.db')),
        DeclareLaunchArgument('foxglove', default_value='true'),
        DeclareLaunchArgument('people', default_value='true'),
        DeclareLaunchArgument('web', default_value='true'),
        DeclareLaunchArgument('web_port', default_value='8080'),
        DeclareLaunchArgument('ir_depth_only', default_value='true'),
        DeclareLaunchArgument('ir_intensity', default_value='4'),
        DeclareLaunchArgument('nav', default_value='true', description='Run Nav2'),
        DeclareLaunchArgument('driver', default_value='true', description='Run the Roomba base driver'),
        DeclareLaunchArgument('driver_params', default_value=PathJoinSubstitution(
            [bringup, 'config', 'roomba_driver_robot.yaml'])),
        DeclareLaunchArgument('nav_params', default_value=PathJoinSubstitution(
            [bringup, 'config', 'nav2_robot.yaml'])),
        # base_link -> camera_link (camera_link is centred between the lenses, x forward, z up).
        DeclareLaunchArgument('cam_x', default_value='0.153', description='m ahead of the Roomba centre'),
        DeclareLaunchArgument('cam_y', default_value='0.0', description='m to the left'),
        DeclareLaunchArgument('cam_z', default_value='0.13',
                              description='m above the floor (tape measure, floor to lens centres)'),
        DeclareLaunchArgument('cam_pitch', default_value='0.0',
                              description='rad, positive = camera tilted down'),
        DeclareLaunchArgument('floor_height', default_value='0.11',
                              description='camera height above the floor as seen in the depth data '
                                          '(calibrated: ~/roomba_tools/floor_cal.py read 0.106-0.113 m; '
                                          'the tape says 0.13 because depth runs ~15% short)'),
    ]

    mount = Node(
        package='tf2_ros', executable='static_transform_publisher', name='camera_mount_tf',
        arguments=['--x', lc('cam_x'), '--y', lc('cam_y'), '--z', lc('cam_z'),
                   '--roll', '0', '--pitch', lc('cam_pitch'), '--yaw', '0',
                   '--frame-id', 'base_link', '--child-frame-id', 'camera_link'],
    )

    vslam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([bringup, 'launch', 'vslam.launch.py'])),
        launch_arguments=[(a, lc(a)) for a in VSLAM_ARGS] + [
            ('platform', 'robot'),
            ('base_frame', 'base_link'),
        ],
    )

    driver = Node(
        package='roomba_driver', executable='roomba_driver', name='roomba_driver', output='screen',
        parameters=[lc('driver_params')],
        condition=IfCondition(lc('driver')),
    )

    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([bringup, 'launch', 'nav.launch.py'])),
        launch_arguments=[
            ('params_file', lc('nav_params')),
            ('base_frame', 'base_link'),
            ('robot_mode', 'true'),
            ('fallback_camera_height_m', lc('floor_height')),
            # Rigid mount: pin the floor instead of re-detecting it every frame (the floor
            # often fills too little of the view to detect, e.g. facing furniture on carpet).
            ('fixed_camera_height_m', lc('floor_height')),
        ],
        condition=IfCondition(lc('nav')),
    )

    return LaunchDescription(args + [mount, vslam, driver, nav])
