"""Nav2 navigation, for the "human robot" test or the real Roomba.

Hand-held test (a person carries the camera and follows on-screen instructions); run alongside
vslam.launch.py, which provides map -> odom -> camera_link and the depth/IMU:
  ros2 launch roomba_bringup nav.launch.py
On the robot, robot.launch.py includes this file with the robot's frames and nav2_robot.yaml.

Pipeline: floor_obstacles (depth + IMU) -> costmaps -> planner + RPP controller
          -> cmd_vel_nav -> collision_monitor (stop zone) -> cmd_vel -> nav_coach -> web app
          (and, on the robot, cmd_vel -> roomba_driver). On the robot, explorer drives to the
          edges of the map on request (web app Explore button).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

NAV_NODES = ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator', 'collision_monitor']


def generate_launch_description():
    params = LaunchConfiguration('params_file')
    args = [
        DeclareLaunchArgument('params_file', default_value=PathJoinSubstitution(
            [FindPackageShare('roomba_bringup'), 'config', 'nav2_human.yaml'])),
        DeclareLaunchArgument('fallback_camera_height_m', default_value='0.4',
                              description='Used only when no floor is visible'),
        DeclareLaunchArgument('fixed_camera_height_m', default_value='0.0',
                              description='Rigid mount: camera height above the floor (0 = estimate)'),
        DeclareLaunchArgument('base_frame', default_value='camera_link',
                              description='Robot body frame (base_link on the robot)'),
        DeclareLaunchArgument('robot_mode', default_value='false',
                              description='true: the robot drives itself (status wording instead of '
                                          'instructions for a person)'),
    ]

    nodes = [
        Node(package='roomba_perception', executable='floor_obstacles', name='floor_obstacles',
             output='screen',
             parameters=[{
                 'fallback_camera_height_m': ParameterValue(LaunchConfiguration('fallback_camera_height_m'),
                                                            value_type=float),
                 'fixed_camera_height_m': ParameterValue(LaunchConfiguration('fixed_camera_height_m'),
                                                         value_type=float),
                 # On the robot, don't mark obstacles during fast in-place turns: the pose lags
                 # and the points would be smeared across the costmap.
                 'pause_turn_rate_rps': ParameterValue(
                     PythonExpression(["0.5 if '", LaunchConfiguration('robot_mode'), "' == 'true' else 0.0"]),
                     value_type=float),
             }]),
        Node(package='nav2_controller', executable='controller_server', name='controller_server', output='screen',
             parameters=[params], remappings=[('cmd_vel', 'cmd_vel_nav')]),
        Node(package='nav2_planner', executable='planner_server', name='planner_server', output='screen',
             parameters=[params]),
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server', output='screen',
             parameters=[params], remappings=[('cmd_vel', 'cmd_vel_nav')]),
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator', output='screen',
             parameters=[params], condition=UnlessCondition(LaunchConfiguration('robot_mode'))),
        # The robot can't see behind it: its tree backs up only 10 cm, at most once per goal.
        Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator', output='screen',
             parameters=[params, {'default_nav_to_pose_bt_xml': PathJoinSubstitution(
                 [FindPackageShare('roomba_bringup'), 'config', 'navigate_robot_bt.xml'])}],
             condition=IfCondition(LaunchConfiguration('robot_mode'))),
        Node(package='nav2_collision_monitor', executable='collision_monitor', name='collision_monitor',
             output='screen', parameters=[params]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation',
             output='screen', parameters=[{'autostart': True, 'node_names': NAV_NODES, 'bond_timeout': 10.0}]),
        Node(package='roomba_perception', executable='nav_coach', name='nav_coach', output='screen',
             parameters=[{
                 'base_frame': LaunchConfiguration('base_frame'),
                 'robot_mode': ParameterValue(LaunchConfiguration('robot_mode'), value_type=bool),
             }]),
        # Frontier exploration (robot only): started and stopped from the web app.
        Node(package='roomba_perception', executable='explorer', name='explorer', output='screen',
             parameters=[{'base_frame': LaunchConfiguration('base_frame')}],
             condition=IfCondition(LaunchConfiguration('robot_mode'))),
    ]
    return LaunchDescription(args + nodes)
