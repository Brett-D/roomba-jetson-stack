# Closed loop on the PC: Isaac Sim (Windows, GUI) drives the robot's real stack in WSL (cuVSLAM + EKF, nvblox,
# RTAB-Map mapping into a test database, Nav2, Rust nodes, explorer, person detector, web app on :18098).
#
#   powershell -ExecutionPolicy Bypass -File twin\isaac\run_closed_loop.ps1                 # start (or switch a running sim over)
#   powershell -ExecutionPolicy Bypass -File twin\isaac\run_closed_loop.ps1 -Stop           # graceful stop, results copied to E:\twin\runs\closedloop
#   ... -People 2 -Engine yolo11s_synth_384x640_fp16.engine -Record -Name my_run
#
# Order: WSL side first (Zenoh router, Isaac OI bridge + the real roomba_driver, robot.launch.py, run monitor), then
# Isaac Sim (rmw_zenoh client of tcp/127.0.0.1:7447, ROS_DOMAIN_ID 60). Either order works with Zenoh.
# Isaac Sim's GUI needs the RTSS (MSI Afterburner / RivaTuner) Vulkan layer disabled for its process: the layer hooks
# the swapchain and the NVIDIA Vulkan driver crashed ~3 s after start (integer divide by zero in nvoglv64.dll).
# DISABLE_RTSS_LAYER / DISABLE_VK_LAYER_VALVE_steam_* below (and in run_closed_loop.py) affect only this process.
param(
    [switch]$Stop,
    [string]$Name = '',
    [int]$People = 2,
    [string]$Engine = 'yolo11n_384x640_fp16.engine',
    [switch]$Record,
    [switch]$Headless,
    # The explorer's goal/return timeouts are wall clock; the sim runs at ~0.25x real time, so scale them up.
    [double]$WallTimeoutScale = 5,
    # Skip the 360 deg look-around before exploring (it works around an explorer finding, see README)
    [switch]$NoLookAround,
    # No interventions: by default a robot stuck for 40 s sim is reversed 0.3 m (logged as an assist, see README)
    [switch]$NoAssist,
    # Don't add the stairwell keep-out zone (as drawn in the web app) before exploring: the stack then drives into the railing
    [switch]$NoStairKeepout
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'env.ps1')
$wslDir = '/mnt/' + $here.Substring(0, 1).ToLower() + $here.Substring(2).Replace('\', '/') + '/wsl'
$run = 'E:\twin\run'
$ctl = "$run\isaac_control.json"
$pidFile = "$run\isaac_kit.pid"
New-Item -ItemType Directory -Force $run | Out-Null

function Now-Status($msg) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg" | Set-Content E:\twin\isaac_NOW.txt -Encoding utf8 }
function Isaac-Alive { if (Test-Path $pidFile) { $p = Get-Process -Id ([int](Get-Content $pidFile)) -ErrorAction SilentlyContinue; return [bool]$p }; return $false }
function Set-Control($obj) { $obj | ConvertTo-Json -Compress | Set-Content $ctl -Encoding ascii }

if ($Stop) {
    $cur = (wsl -d Ubuntu-24.04 -u root -- bash -c "readlink -f /root/isaacloop/current 2>/dev/null").Trim()
    wsl -d Ubuntu-24.04 -u root -- bash "$wslDir/closed_loop.sh" stop
    if ($cur) {
        $dst = "E:\twin\runs\closedloop\$(Split-Path -Leaf $cur)"
        New-Item -ItemType Directory -Force $dst | Out-Null
        wsl -d Ubuntu-24.04 -u root -- bash -c "cp -r '$cur/monitor' '$cur/logs' '$cur/roomba_driver_isaac.yaml' /mnt/e/twin/runs/closedloop/$(Split-Path -Leaf $cur)/ 2>/dev/null; ls -la '$cur' | head -20"
        "results: $dst"
    }
    if (Isaac-Alive) {
        Set-Control @{ quit = $true; drive = 'stop' }
        $p = Get-Process -Id ([int](Get-Content $pidFile))
        if (-not $p.WaitForExit(60000)) { Stop-Process -Id $p.Id -Force }
    }
    Now-Status 'Closed loop stopped: Isaac Sim window closed, robot stack in WSL stopped, web app (localhost:18098) off.'
    return
}

if (-not $Name) { $Name = 'cl_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
Now-Status "Starting the closed loop ($Name): robot stack in WSL first, then Isaac Sim ..."
$rec = if ($Record) { "E:\twin\videos\raw\$Name" } else { '' }
$dock = @(0.0, 0.0, 0.0)
$alive = Isaac-Alive
if ($alive) {
    # robot back to the dock and parked BEFORE the stack starts: its map frame is the start pose
    Set-Control @{ ros = $true; drive = 'stop'; teleport = $dock; id = $Name; record = '' }
    Start-Sleep -Seconds 3
}
wsl -d Ubuntu-24.04 -u root -- env "WALL_TIMEOUT_SCALE=$WallTimeoutScale" "LOOK_AROUND=$(if ($NoLookAround) { 0 } else { 1 })" "ASSIST=$(if ($NoAssist) { 0 } else { 1 })" "STAIR_KEEPOUT=$(if ($NoStairKeepout) { 0 } else { 1 })" bash "$wslDir/closed_loop.sh" start $Name $Engine
if ($alive) {
    Set-Control @{ ros = $true; drive = 'ros'; teleport = $dock; id = $Name; record = $rec }
} else {
    $env:DISABLE_RTSS_LAYER = '1'
    $env:DISABLE_VK_LAYER_VALVE_steam_overlay_1 = '1'
    $env:DISABLE_VK_LAYER_VALVE_steam_fossilize_1 = '1'
    $env:RMW_IMPLEMENTATION = 'rmw_zenoh_cpp'
    $env:ZENOH_CONFIG_OVERRIDE = 'mode="client";connect/endpoints=["tcp/127.0.0.1:7447"]'
    $env:ROS_DOMAIN_ID = '60'
    Set-Control @{ ros = $true; drive = 'ros' }
    $a = @("$here\scripts\run_closed_loop.py", '--drive', 'ros', '--ros', '--people', "$People", '--viewer', $(if ($Headless) { 'file' } else { 'kit' }))
    if ($rec) { $a += @('--record', $rec) }
    $p = Start-Process -FilePath "$env:TWIN\isaacsim-env\Scripts\python.exe" -ArgumentList $a -WorkingDirectory "$here\scripts" `
        -RedirectStandardOutput "$run\isaac_kit.log" -RedirectStandardError "$run\isaac_kit.err" -PassThru -NoNewWindow
    $p.Id | Set-Content $pidFile
}
"Isaac Sim: window 'Isaac Sim Python 6.2.0' (follow camera); web app: http://localhost:18098 (password in E:\twin\isaac_watch_password.txt)"
"stop with: run_closed_loop.ps1 -Stop"
