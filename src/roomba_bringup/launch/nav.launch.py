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
        # Learned navigation skills (docs/learned-skills.md), off by default; need roomba_rs built with
        # -DLEARNED_SKILLS=ON and the Rust explorer.
        DeclareLaunchArgument('unstuck_skill', default_value='false',
                              description='true: run unstuck_skill and let the explorer call it when stuck'),
        DeclareLaunchArgument('goal_ranking', default_value='rule',
                              description="explorer goal choice: 'rule' or 'learned' (frontier ranker)"),
    ]
    models = PathJoinSubstitution([FindPackageShare('roomba_rs'), 'models'])

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
                 # On the robot, report drop-offs: depth points more than 12 cm below the floor within
                 # 2 m (a stairwell seen through the railing's balusters) put obstacle points at the
                 # edge, where their line of sight crosses the floor plane; before, the drop was
                 # mapped as floor and Nav2 planned through the railing.
                 'drop_min_depth_m': ParameterValue(
                     PythonExpression(["0.12 if '", LaunchConfiguration('robot_mode'), "' == 'true' else 0.0"]),
                     value_type=float),
                 'drop_max_range_m': 2.0,
                 'drop_min_points': 6,
                 # ... 10 cm nearer the robot than the edge, so the planner keeps it 10 cm off the
                 # railing / edge (Isaac run 9: the body rubbed along the railing at the stair head).
                 'drop_edge_margin_m': 0.10,
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
        # The lifecycle manager waits forever for each change_state answer; one lost answer (seen
        # once in ~25 twin launches) left Nav2 inactive ("Navigation is not running"). The guard
        # checks the servers' states after 30 s and brings up whatever isn't active.
        Node(package='roomba_rs', executable='nav2_guard', name='nav2_guard', output='screen',
             parameters=[{'node_names': NAV_NODES, 'startup_timeout_s': 30.0}]),
        # Rust port: the same instructions tick for tick (tools/ab_nav_coach_offline.py), ~1/6 of the CPU.
        Node(package=PythonExpression(["'roomba_rs' if '", LaunchConfiguration('rust_nodes'),
                                       "' == 'true' else 'roomba_perception'"]),
             executable='nav_coach', name='nav_coach', output='screen',
             parameters=[{
                 'base_frame': LaunchConfiguration('base_frame'),
                 'robot_mode': ParameterValue(LaunchConfiguration('robot_mode'), value_type=bool),
             }]),
        # Frontier exploration (robot only): started and stopped from the web app.
        # Rust port: the same goals, status and keep-out marks (tools/ab_explorer_offline.py), ~1/6 of the CPU.
        Node(package=PythonExpression(["'roomba_rs' if '", LaunchConfiguration('rust_nodes'),
                                       "' == 'true' else 'roomba_perception'"]),
             executable='explorer', name='explorer', output='screen',
             # The driver estimates the Ridgid pack's charge from its voltage (battery_gauge: voltage).
             # Bump marks: wider (this robot's right bumper switch is masked, so hits on the right
             # read as left), and also on the front light bumpers, which see railing posts first.
             # Marks leave the robot's footprint + 0.1 m free (a cell diagonal: the mask and costmap
             # grids don't line up; a mark over the robot trapped it); it looks around (4 x 90 deg)
             # at the first spot it reaches (not on the dock) and before giving up. Frontiers longer
             # than 2 m get a goal spot per 2 m piece (a spot behind a stair railing doesn't drop a
             # whole side of the house).
             parameters=[{'base_frame': LaunchConfiguration('base_frame'), 'low_battery_fraction': 0.2,
                          'bump_mark_width_m': 0.5, 'bump_light_signal': 200, 'mark_clear_radius_m': 0.28,
                          'mark_soft_band_m': 0.22, 'look_around_steps': 4, 'look_around_step_deg': 90.0,
                          # Drop-offs floor_obstacles reports (a stair opening the map shows as
                          # floor) keep frontiers and goal spots 0.3 m away.
                          'drop_clear_m': 0.3,
                          # Bumps with a person seen in front are not marked, a mark lasts 60 s
                          # unless the same spot is bumped again, and marks that box it in are removed.
                          'bump_mark_short_ttl_s': 60.0, 'person_bump_dist_m': 1.2, 'trap_area_m2': 1.0,
                          # A mark bumped again is a railing or glass: the robot's centre stays 0.28 m
                          # from it (footprint 0.17 + 0.1 m; keep-out cells aren't inflated, and in
                          # Isaac run 9 it rubbed along the stair railing).
                          'bump_mark_confirmed_margin_m': 0.28,
                          'frontier_split_m': 2.0,
                          'unstuck_skill': ParameterValue(LaunchConfiguration('unstuck_skill'), value_type=bool),
                          'goal_ranking': LaunchConfiguration('goal_ranking'),
                          'goal_ranking_model': PathJoinSubstitution([models, 'explore_ranker.onnx'])}],
             condition=IfCondition(LaunchConfiguration('robot_mode'))),
        # Learned unstuck skill (opt-in): drives through cmd_vel_nav, so the smoother, the collision
        # monitor and the driver's reflexes stay in charge; respawned if its watchdog fires.
        Node(package='roomba_rs', executable='unstuck_skill', name='unstuck_skill', output='screen',
             respawn=True, respawn_delay=2.0,
             parameters=[{'model': PathJoinSubstitution([models, 'unstuck_policy.onnx'])}],
             condition=IfCondition(LaunchConfiguration('unstuck_skill'))),
    ]
    return LaunchDescription(args + nodes)
