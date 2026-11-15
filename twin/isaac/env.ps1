# Dot-source before any twin command: keeps every cache/temp on E:\twin.
$env:TWIN = 'E:\twin'
$env:UV_CACHE_DIR = 'E:\twin\cache\uv'
$env:UV_PYTHON_INSTALL_DIR = 'E:\twin\python'
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:PIP_CACHE_DIR = 'E:\twin\cache\pip'
$env:TEMP = 'E:\twin\tmp'; $env:TMP = 'E:\twin\tmp'
$env:TORCH_HOME = 'E:\twin\cache\torch'
$env:HF_HOME = 'E:\twin\cache\hf'
$env:YOLO_CONFIG_DIR = 'E:\twin\cache\ultralytics'
$env:CUDA_CACHE_PATH = 'E:\twin\cache\nv_compute'
$env:WARP_CACHE_PATH = 'E:\twin\cache\warp'
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:PATH = 'E:\twin\tools\uv;' + $env:PATH
foreach ($d in 'E:\twin\tmp','E:\twin\cache\uv','E:\twin\cache\pip','E:\twin\cache\torch','E:\twin\cache\hf','E:\twin\cache\ultralytics','E:\twin\cache\nv_compute','E:\twin\cache\warp') { New-Item -ItemType Directory -Force $d | Out-Null }
$env:UV_PYTHON_BIN_DIR = 'E:\twin\python\bin'
