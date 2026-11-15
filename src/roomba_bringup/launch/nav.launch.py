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

NAV_NODES = ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator', 'velocity_smoother',
             'collision_monitor']


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
        DeclareLaunchArgument('rust_nodes', default_value='true',
                              description='Use the Rust ports (package roomba_rs) of nodes that have one; '
                                          'false = the original Python nodes'),
        DeclareLaunchArgument('robot_mode', default_value='false',
                              description='true: the robot drives itself (status wording instead of '
                                          'instructions for a person)'),
    ]

    nodes = [
        # Rust port: same topics, parameters and output, ~1/7 of the CPU (A/B on a recorded drive).
        Node(package=PythonExpression(["'roomba_rs' if '", LaunchConfiguration('rust_nodes'),
                                       "' == 'true' else 'roomba_perception'"]),
             executable='floor_obstacles', name='floor_obstacles',
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
                 # On the robot, mark low edges (a railing's base rail, thresholds) from 4.5 cm:
                 # it once climbed a base rail that the 6 cm default ignored.
                 'min_obstacle_height_m': ParameterValue(
                     PythonExpression(["0.045 if '", LaunchConfiguration('robot_mode'), "' == 'true' else 0.06"]),
                     value_type=float),
                 # ...and drop stray depth speckles (they showed up as random obstacles mid-room).
                 'min_cell_points': ParameterValue(
                     PythonExpression(["4 if '", LaunchConfiguration('robot_mode'), "' == 'true' else 1"]),
                     value_type=int),
                 'max_range_m': ParameterValue(
                     PythonExpression(["2.5 if '", LaunchConfiguration('robot_mode'), "' == 'true' else 4.0"]),
                     value_type=float),
             }]),
        # Joins dotted obstacles in RTAB-Map's grid (railing posts) into solid walls for the costmaps.
        # Rust port: the same grid cell for cell (tools/ab_map_gap_filler_offline.py), ~20x faster.
        Node(package=PythonExpression(["'roomba_rs' if '", LaunchConfiguration('rust_nodes'),
                                       "' == 'true' else 'roomba_perception'"]),
             executable='map_gap_filler', name='map_gap_filler', output='screen'),
        Node(package='nav2_controller', executable='controller_server', name='controller_server', output='screen',
             parameters=[params], remappings=[('cmd_vel', 'cmd_vel_nav')]),
        Node(package='nav2_planner', executable='planner_server', name='planner_server', output='screen',
             parameters=[params]),
        Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server', output='screen',
             parameters=[params], remappings=[('cmd_vel', 'cmd_vel_nav')]),
        # cmd_vel_nav (controller, behaviours, web drive pad) -> velocity_smoother -> cmd_vel_smoothed
        # -> collision_monitor -> cmd_vel. With the human-mode params the smoother's output is unused.
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', name='velocity_smoother',
             output='screen', parameters=[params],
             remappings=[('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel_smoothed')]),
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
             # The driver estimates the Ridgid pack's charge from its voltage (battery_gauge: voltage).
             # Bump marks: wider (this robot's right bumper switch is masked, so hits on the right
             # read as left), and also on the front light bumpers, which see railing posts first.
             parameters=[{'base_frame': LaunchConfiguration('base_frame'), 'low_battery_fraction': 0.2,
                          'bump_mark_width_m': 0.5, 'bump_light_signal': 200}],
             condition=IfCondition(LaunchConfiguration('robot_mode'))),
    ]
    return LaunchDescription(args + nodes)
