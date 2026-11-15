"""Handheld visual SLAM test: MYNT D1000 (RGB-D + IMU) -> RTAB-Map, viewable over foxglove_bridge.

Usage:
  ros2 launch roomba_bringup handheld_slam.launch.py
  ros2 launch roomba_bringup handheld_slam.launch.py localization:=true   # reuse saved map
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    localization = LaunchConfiguration('localization')
    database_path = LaunchConfiguration('database_path')

    args = [
        DeclareLaunchArgument('localization', default_value='false',
                              description='true = localize in an existing map instead of mapping'),
        DeclareLaunchArgument('database_path',
                              default_value=os.path.expanduser('~/maps/handheld.db')),
        DeclareLaunchArgument('foxglove', default_value='true'),
        # Stopping: SIGINT, SIGTERM 5 s later, SIGKILL this much later still; RTAB-Map writes its
        # map database while it shuts down (launch's default 5 s could kill it mid-write).
        DeclareLaunchArgument('sigkill_timeout', default_value='85',
                              description='s from SIGTERM to SIGKILL when stopping (RTAB-Map saves its map)'),
    ]

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare('mynteye_d1000_driver'), '/launch/d1000.launch.py']),
    )

    imu_filter = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        namespace='camera',
        parameters=[{'use_mag': False, 'publish_tf': False, 'world_frame': 'enu'}],
        remappings=[('imu/data_raw', '/camera/imu/data_raw'), ('imu/data', '/camera/imu/data')],
    )

    # Mapping deletes the previous database on start; localization reuses it.
    # Images already arrive at 640x360 from the driver, so no extra decimation here.
    rtabmap_args = PythonExpression([
        "'' if '", localization, "' == 'true' else '--delete_db_on_start'",
        " + ' --Grid/RangeMax 4.0 --Grid/MaxObstacleHeight 1.0'",
    ])

    rtabmap = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare('rtabmap_launch'), '/launch/rtabmap.launch.py']),
        launch_arguments={
            'namespace': 'rtabmap',
            'rgb_topic': '/camera/left/image_rect_color',
            'depth_topic': '/camera/depth/image_rect_raw',
            'camera_info_topic': '/camera/left/camera_info',
            'frame_id': 'camera_link',
            'imu_topic': '/camera/imu/data',
            'wait_imu_to_init': 'true',
            # The driver stamps each interleaved depth frame with its color frame's time,
            # so color/depth pair exactly; odometry/IMU still use approximate sync.
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.05',
            'rgbd_sync': 'true',
            'approx_rgbd_sync': 'false',
            'topic_queue_size': '3',
            'visual_odometry': 'true',
            # ResetCountdown: if tracking is lost, restart odometry on the next frame; RTAB-Map
            # starts a new map session and re-links it on the next loop closure.
            'odom_args': '--Odom/ResetCountdown 1 --Vis/MaxFeatures 600 --OdomF2M/MaxSize 1000',
            'rtabmap_args': rtabmap_args,
            'localization': localization,
            'database_path': database_path,
            'rtabmap_viz': 'false',
            'rviz': 'false',
        }.items(),
    )

    # Compressed left image for remote viewing (separate process keeps cv_bridge out of the driver).
    republish = Node(
        package='image_transport',
        executable='republish',
        name='left_republish',
        namespace='camera',
        parameters=[{'in_transport': 'raw', 'out_transport': 'compressed'}],
        remappings=[('in', '/camera/left/image_rect_color'), ('out/compressed', '/camera/left/view/compressed')],
        condition=IfCondition(LaunchConfiguration('foxglove')),
    )

    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{
            'port': 8765,
            'topic_whitelist': [
                '/tf', '/tf_static', '/camera/left/view/compressed', '/camera/left/camera_info',
                '/camera/imu/data', '/rtabmap/map', '/rtabmap/odom', '/rtabmap/mapPath',
                '/rtabmap/cloud_map', '/rtabmap/info', '/rtabmap/odom_info', '/rtabmap/odom_last_frame',
                '/rosout', '/diagnostics',
            ],
        }],
        condition=IfCondition(LaunchConfiguration('foxglove')),
    )

    return LaunchDescription(args + [camera, imu_filter, rtabmap, republish, foxglove])
