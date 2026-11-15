# Dot-source to use Isaac Sim's bundled ROS 2 Jazzy (rclpy + messages) from the venv python, e.g. for
# ros_check.py. Same settings must be in effect for the simulator process (run_twin_ros.py sets them).
. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'env.ps1')
$ros = "$env:TWIN\isaacsim-env\Lib\site-packages\isaacsim\exts\isaacsim.ros2.core\jazzy"
$env:ROS_DISTRO = 'jazzy'
$env:RMW_IMPLEMENTATION = 'rmw_fastrtps_cpp'
$env:ROS_DOMAIN_ID = '69'
# Phase 1 is PC-local: keep DDS discovery on this machine (no traffic to the Jetson / LAN).
$env:ROS_AUTOMATIC_DISCOVERY_RANGE = 'LOCALHOST'
$env:PATH = "$ros\lib;" + $env:PATH
# Appended (not prepended) to sys.path by the scripts: the bundled tree carries its own numpy that
# must not shadow the venv's.
$env:TWIN_ROS_PY = "$ros\rclpy"
$env:AMENT_PREFIX_PATH = $ros
