"""PC smoke-test stack: the robot's Nav2 (nav.launch.py, robot mode, nav2_robot.yaml, Rust nodes)
on a static map, ignore_zones for the keep-out filter, a kinematic base stand-in (kinsim.py) and
optionally the Rust roomba_driver on tools/fake_roomba.py (driver_params:=<generated yaml>)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    bringup = get_package_share_directory('roomba_bringup')
    work = LaunchConfiguration('work')
    return LaunchDescription([
        DeclareLaunchArgument('work', default_value=HERE),
        DeclareLaunchArgument('driver_params', default_value=''),
        DeclareLaunchArgument('nav_params', default_value=os.path.join(bringup, 'config', 'nav2_robot.yaml')),
        Node(package='tf2_ros', executable='static_transform_publisher', name='map_odom_tf',
             arguments=['--frame-id', 'map', '--child-frame-id', 'odom']),
        Node(package='tf2_ros', executable='static_transform_publisher', name='camera_mount_tf',
             arguments=['--x', '0.153', '--z', '0.13', '--frame-id', 'base_link', '--child-frame-id', 'camera_link']),
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             parameters=[{'yaml_filename': PythonExpression(["'", work, "' + '/map.yaml'"]),
                          'topic_name': '/rtabmap/map', 'frame_id': 'map'}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_map',
             parameters=[{'autostart': True, 'node_names': ['map_server']}]),
        Node(package='roomba_rs', executable='ignore_zones', name='ignore_zones', namespace='perception',
             parameters=[{'zones_file': PythonExpression(["'", work, "' + '/zones.json'"]), 'clear_on_start': True}]),
        ExecuteProcess(cmd=['python3', os.path.join(HERE, 'kinsim.py')], name='kinsim'),
        Node(package='roomba_rs', executable='roomba_driver', name='roomba_driver', output='screen',
             parameters=[LaunchConfiguration('driver_params')],
             condition=IfCondition(PythonExpression(["'", LaunchConfiguration('driver_params'), "' != ''"]))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', 'nav.launch.py')),
            launch_arguments={
                'params_file': LaunchConfiguration('nav_params'),
                'base_frame': 'base_link', 'robot_mode': 'true', 'rust_nodes': 'true',
                'fallback_camera_height_m': '0.11', 'fixed_camera_height_m': '0.11',
            }.items()),
    ])
