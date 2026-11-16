param(
    [ValidateSet('chat', 'image')][string]$Probe = 'chat',
    [ValidateSet('rmw_zenoh_cpp', 'rmw_fastrtps_cpp')][string]$Rmw = 'rmw_zenoh_cpp',
    [string]$Connect = 'tcp/127.0.0.1:7447',
    [string]$FastddsProfile = '',
    [int]$Secs = 20,
    [string]$Domain = '42',
    [string]$Twin = 'E:\twin'
)
# Windows side of the WSL <-> Windows ROS 2 probes (tools/pc_wsl/dds_probe.py, img_probe.py), run with
# Isaac Sim's bundled ROS 2 Jazzy (the twin's python), without starting Isaac Sim. See docs/pc-wsl.md.
#   chat:  publishes /pc_wsl_probe/win, counts /pc_wsl_probe/wsl (WSL side: dds_probe.py wsl win <secs>)
#   image: publishes 848x480 bgr8 at 30 Hz on /pc_wsl_probe/image (WSL side: img_probe.py sub <secs>)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$ros = "$Twin\isaacsim-env\Lib\site-packages\isaacsim\exts\isaacsim.ros2.core\jazzy"
$env:ROS_DISTRO = 'jazzy'
$env:RMW_IMPLEMENTATION = $Rmw
$env:ROS_DOMAIN_ID = $Domain
Remove-Item Env:ROS_AUTOMATIC_DISCOVERY_RANGE -ErrorAction SilentlyContinue
$env:PATH = "$ros\lib;" + $env:PATH
$env:TWIN_ROS_PY = "$ros\rclpy"
$env:AMENT_PREFIX_PATH = $ros
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:ROS_LOG_DIR = Join-Path $env:TEMP 'pc_wsl_probe_log'
if ($Rmw -eq 'rmw_zenoh_cpp') { $env:ZENOH_CONFIG_OVERRIDE = "mode=`"client`";connect/endpoints=[`"$Connect`"]" }
if ($FastddsProfile) { $env:FASTRTPS_DEFAULT_PROFILES_FILE = $FastddsProfile }
$py = "$Twin\isaacsim-env\Scripts\python.exe"
if ($Probe -eq 'chat') { & $py "$here\dds_probe.py" win wsl $Secs 2>&1 | Where-Object { $_ -notmatch 'deprecated' } }
else { & $py "$here\img_probe.py" pub $Secs 2>&1 | Where-Object { $_ -notmatch 'deprecated' } }
