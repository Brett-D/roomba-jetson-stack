"""The Roomba robot: camera + VSLAM + Roomba base driver + Nav2, all in one launch.

  ros2 launch roomba_bringup robot.launch.py mode:=mapping         # drive around to build a new map
  ros2 launch roomba_bringup robot.launch.py                       # reuse the saved map (auto)
  ros2 launch roomba_bringup robot.launch.py nav:=false            # VSLAM + driver only
  ros2 launch roomba_bringup robot.launch.py sounds:=false         # no event sounds on the Roomba

Pipelines (A is the default; any mix of the three choices works):
  A  odometry:=basalt  fusion:=watchdog  obstacles:=floor     Basalt VIO (CPU), vslam_watchdog
                                                               TF, floor_obstacles voxel costmaps
  B  odometry:=cuvslam fusion:=ekf       obstacles:=nvblox    cuVSLAM (GPU), robot_localization
                                                               EKF (wheels + gyro + VIO), nvblox
                                                               costmaps (GPU)
Replays (scripts/compare_pipelines.sh) add camera:=false driver:=false use_sim_time:=true.

Frames: map (RTAB-Map) -> odom (vslam_watchdog: VIO, wheel-bridged; or the EKF) -> base_link (Roomba centre
on the floor, x forward) -> camera_link (static mount below) -> optical / IMU frames (camera driver).
roomba_driver drives the wheels from /cmd_vel (Nav2's collision monitor, or the web app's drive pad)
and publishes /wheel/odom, /battery_state and /roomba/sensors. rust_nodes:=false runs the original
Python nodes (driver included) instead of their Rust ports in roomba_rs and roomba_driver_rs.

Camera mount defaults: centred, facing forward, level, lens 0.153 m ahead of the Roomba's centre
(1.7 cm behind the front bumper edge) and 0.13 m above the floor (tape measure). In the depth
data the floor is ~0.11 m below the lens (depth runs ~15% short); floor_obstacles uses that
calibrated floor_height directly because the mount is rigid.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# Passed through to vslam.launch.py unchanged.
VSLAM_ARGS = ['mode', 'odometry', 'database_path', 'foxglove', 'people', 'web', 'web_port',
              'ir_depth_only', 'ir_intensity', 'ground_constraint', 'fusion', 'camera', 'use_sim_time',
              'rust_nodes']


def _check_args(context):
    obstacles = LaunchConfiguration('obstacles').perform(context)
    if obstacles not in ('floor', 'nvblox'):
        raise RuntimeError(f"obstacles must be floor or nvblox (got '{obstacles}')")
    odometry = LaunchConfiguration('odometry').perform(context)
    if odometry not in ('basalt', 'cuvslam'):
        raise RuntimeError(f"odometry must be basalt or cuvslam (got '{odometry}')")
    return [LogInfo(msg=f"[robot] odometry={odometry} fusion={LaunchConfiguration('fusion').perform(context)} "
                        f"obstacles={obstacles} nav_params={LaunchConfiguration('nav_params').perform(context)}")]


def generate_launch_description():
    bringup = FindPackageShare('roomba_bringup')
    bringup_share = get_package_share_directory('roomba_bringup')
    lc = LaunchConfiguration
    args = [
        DeclareLaunchArgument('mode', default_value='auto',
                              description='auto (reuse saved map if present), mapping (new map), localization'),
        DeclareLaunchArgument('odometry', default_value='basalt',
                              description='basalt (CPU) or cuvslam (GPU, Isaac ROS Visual SLAM)'),
        DeclareLaunchArgument('fusion', default_value='watchdog',
                              description='watchdog (VIO filtered + wheel-bridged by vslam_watchdog) or ekf '
                                          '(robot_localization: wheels + gyro + VIO, config/ekf_robot.yaml)'),
        DeclareLaunchArgument('obstacles', default_value='floor',
                              description='Costmap obstacles: floor (floor_obstacles + voxel layer) or nvblox '
                                          '(GPU TSDF/ESDF, config/nvblox_robot.yaml). The collision monitor '
                                          'always uses floor_obstacles.'),
        DeclareLaunchArgument('camera', default_value='true',
                              description='false: no camera driver (replaying a recorded bag)'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true when replaying a bag with --clock (compare_pipelines.sh)'),
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
        DeclareLaunchArgument('sounds', default_value='true',
                              description='Event sounds on the Roomba (roomba_rs sound_events): hello, uh-oh on '
                                          'bumps, a jingle when exploring finishes; false = none'),
        DeclareLaunchArgument('rust_nodes', default_value='true',
                              description='Use the Rust ports (package roomba_rs) of nodes that have one, and the '
                                          'Rust Roomba driver (roomba_driver_rs); false = the original '
                                          'Python nodes'),
        DeclareLaunchArgument('driver_params', default_value=PathJoinSubstitution(
            [bringup, 'config', 'roomba_driver_robot.yaml'])),
        DeclareLaunchArgument('nav_params', default_value=PythonExpression([
            "'", bringup_share, "/config/' + ('nav2_robot_nvblox.yaml' if '", lc('obstacles'),
            "' == 'nvblox' else 'nav2_robot.yaml')"])),
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

    # Rust driver (roomba_driver_rs, in the roomba-serial-ros2 submodule): same topics, services,
    # parameters and serial bytes as the Python roomba_driver, ~1/10 of the CPU.
    driver = Node(
        package=PythonExpression(["'roomba_driver_rs' if '", lc('rust_nodes'), "' == 'true' else 'roomba_driver'"]),
        executable='roomba_driver', name='roomba_driver', output='screen',
        parameters=[lc('driver_params')],
        condition=IfCondition(lc('driver')),
    )

    # Event sounds (sounds:=true): "hello" when the driver first reports Safe/Full mode, "uh-oh" on a
    # bumper hit, a jingle when an exploration run finishes. Rust only, but it just uses the driver's
    # roomba/define_song service and topics, so it runs with the Python driver (rust_nodes:=false) too.
    # Never changes the OI mode. Mute it live with: touch ~/.config/roomba/sound_muted
    sounds = Node(
        package='roomba_rs', executable='sound_events', name='sound_events', output='screen',
        condition=IfCondition(PythonExpression(["'", lc('sounds'), "' == 'true' and '", lc('driver'), "' == 'true'"])),
    )

    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([bringup, 'launch', 'nav.launch.py'])),
        launch_arguments=[
            ('params_file', lc('nav_params')),
            ('base_frame', 'base_link'),
            ('robot_mode', 'true'),
            ('rust_nodes', lc('rust_nodes')),
            ('fallback_camera_height_m', lc('floor_height')),
            # Rigid mount: pin the floor instead of re-detecting it every frame (the floor
            # often fills too little of the view to detect, e.g. facing furniture on carpet).
            ('fixed_camera_height_m', lc('floor_height')),
        ],
        condition=IfCondition(lc('nav')),
    )

    # GPU obstacle mapping for the costmaps (obstacles:=nvblox): depth -> TSDF -> 2D ESDF slice.
    nvblox = Node(
        package='nvblox_ros', executable='nvblox_node', name='nvblox_node', output='screen',
        parameters=[PathJoinSubstitution([bringup, 'config', 'nvblox_robot.yaml'])],
        remappings=[
            ('camera_0/depth/image', '/camera/depth/image_rect_raw'),
            ('camera_0/depth/camera_info', '/camera/depth/camera_info'),
        ],
        condition=IfCondition(PythonExpression(["'", lc('obstacles'), "' == 'nvblox'"])),
    )

    sim_time = SetParameter(name='use_sim_time', value=ParameterValue(lc('use_sim_time'), value_type=bool))
    return LaunchDescription(args + [OpaqueFunction(function=_check_args), sim_time,
                                     mount, vslam, driver, sounds, nvblox, nav])
