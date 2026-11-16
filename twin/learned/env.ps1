# Dot-source before learned-skills work: caches on E:\twin, own temp, own venv.
$env:UV_CACHE_DIR = 'E:\twin\cache\uv'
$env:UV_PYTHON_INSTALL_DIR = 'E:\twin\python'
$env:UV_PYTHON_PREFERENCE = 'only-managed'
$env:UV_PROJECT_ENVIRONMENT = 'E:\twin\isaaclab-env'
$env:PIP_CACHE_DIR = 'E:\twin\cache\pip'
$env:TEMP = 'E:\twin\learned\tmp'; $env:TMP = 'E:\twin\learned\tmp'
$env:TORCH_HOME = 'E:\twin\cache\torch'
$env:HF_HOME = 'E:\twin\cache\hf'
$env:CUDA_CACHE_PATH = 'E:\twin\cache\nv_compute'
$env:WARP_CACHE_PATH = 'E:\twin\learned\cache\warp'
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
$env:PYTHONUTF8 = '1'
$env:PATH = 'E:\twin\tools\uv;' + $env:PATH
foreach ($d in 'E:\twin\learned\tmp', 'E:\twin\learned\cache\warp') { New-Item -ItemType Directory -Force $d | Out-Null }
