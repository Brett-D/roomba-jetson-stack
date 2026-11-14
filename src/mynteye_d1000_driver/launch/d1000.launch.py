from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument('camera_namespace', default_value='camera'),
        DeclareLaunchArgument('ir_intensity', default_value='4'),
        DeclareLaunchArgument('ir_depth_only', default_value='true',
                              description='IR projector only on depth frames (clean color, 15 Hz each)'),
        DeclareLaunchArgument('publish_points', default_value='false'),
        DeclareLaunchArgument('points_decimation', default_value='4'),
        DeclareLaunchArgument('output_scale', default_value='0.5',
                              description='Scale of published images (0.5 -> 640x360)'),
        DeclareLaunchArgument('enable_imu', default_value='true'),
        DeclareLaunchArgument('publish_tf', default_value='true'),
    ]

    node = Node(
        package='mynteye_d1000_driver',
        executable='d1000_node',
        name='mynteye_d1000',
        namespace=LaunchConfiguration('camera_namespace'),
        output='screen',
        parameters=[{
            'ir_intensity': LaunchConfiguration('ir_intensity'),
            'ir_depth_only': LaunchConfiguration('ir_depth_only'),
            'publish_points': LaunchConfiguration('publish_points'),
            'points_decimation': LaunchConfiguration('points_decimation'),
            'output_scale': LaunchConfiguration('output_scale'),
            'enable_imu': LaunchConfiguration('enable_imu'),
            'publish_tf': LaunchConfiguration('publish_tf'),
        }],
    )

    return LaunchDescription(args + [node])
