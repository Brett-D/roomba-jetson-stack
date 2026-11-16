# Open the live swarm viewer (swarm_view.py): 1536 simulated Roombas in Isaac Lab's Newton viewer.
#   powershell -ExecutionPolicy Bypass -File twin\learned\viz\swarm_view.ps1 [swarm_view.py arguments]
# Uses the Isaac Lab venv and caches of twin\learned\env.ps1; RTSS / Steam overlay layers off (they
# crash windowed GPU apps on this PC).
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here '..\env.ps1')
$env:DISABLE_RTSS_LAYER = '1'
$env:DISABLE_VK_LAYER_VALVE_steam_overlay_1 = '1'
$env:DISABLE_VK_LAYER_VALVE_steam_fossilize_1 = '1'
& 'E:\twin\isaaclab-env\Scripts\python.exe' (Join-Path $here 'swarm_view.py') @args
