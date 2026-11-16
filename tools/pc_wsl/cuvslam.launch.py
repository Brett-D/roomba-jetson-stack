from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    vslam = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'num_cameras': 2,
            'rectified_images': True,
            'tracking_mode': 0,
            'enable_imu_fusion': False,
            'base_frame': 'base_link',
            'odom_frame': 'vslam_odom',
            'map_frame': 'vslam_map',
            'publish_odom_to_base_tf': False,
            'publish_map_to_odom_tf': False,
            'camera_optical_frames': ['left_optical', 'right_optical'],
            'image_qos': 'DEFAULT',
            'verbosity': 1,
        }],
        remappings=[
            ('visual_slam/image_0', '/synth/left/image_rect'),
            ('visual_slam/camera_info_0', '/synth/left/camera_info'),
            ('visual_slam/image_1', '/synth/right/image_rect'),
            ('visual_slam/camera_info_1', '/synth/right/camera_info'),
        ])
    return LaunchDescription([ComposableNodeContainer(
        name='vslam_container', namespace='', package='rclcpp_components',
        executable='component_container_mt', composable_node_descriptions=[vslam], output='screen')])
