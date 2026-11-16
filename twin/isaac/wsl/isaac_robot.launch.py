"""robot.launch.py for the Isaac Sim closed loop, unchanged, plus one sim-side workaround (launch argument
wall_timeout_scale, default 1 = off) that multiplies the explorer's timeouts by k. Up to main 4650dca the explorer
measured them on the wall clock, so at a real-time factor r < 1 a 60 s goal timeout fired after 60 * r s of simulated
driving (k ~ 1/r fixed that). Since cd218c6 it uses the node clock (sim time): leave k = 1. Every other argument is
passed through to robot.launch.py.

  ros2 launch /mnt/e/.../twin/isaac/wsl/isaac_robot.launch.py wall_timeout_scale:=4 <robot.launch.py arguments>
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetParameter

# explorer parameters measured on the wall clock (src/roomba_rs/src/explorer.rs Params) and their defaults
EXPLORER_WALL_TIMEOUTS = {'goal_timeout_s': 60.0, 'goal_response_timeout_s': 10.0, 'return_timeout_s': 300.0,
                          'finish_confirm_s': 20.0, 'look_around_timeout_s': 20.0, 'bump_mark_ttl_s': 600.0}


def _scaled(context):
    k = float(LaunchConfiguration('wall_timeout_scale').perform(context))
    if k == 1.0:
        return []
    return [SetParameter(name=n, value=v * k) for n, v in EXPLORER_WALL_TIMEOUTS.items()]


def generate_launch_description():
    robot = IncludeLaunchDescription(PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory('roomba_bringup'), 'launch', 'robot.launch.py')))
    return LaunchDescription([
        DeclareLaunchArgument('wall_timeout_scale', default_value='1.0'),
        OpaqueFunction(function=_scaled),
        robot,
    ])
