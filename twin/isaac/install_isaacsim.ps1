# Install Isaac Sim 6.2.0 (pip) + an eval env on E:\twin, keeping every cache on E:.
# Run from Windows PowerShell or pwsh:  powershell -ExecutionPolicy Bypass -File install_isaacsim.ps1
# Installing Isaac Sim means accepting the NVIDIA Omniverse License Agreement (OMNI_KIT_ACCEPT_EULA=YES).
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $here 'env.ps1')

# uv (standalone binary, no installer, nothing on PATH permanently)
if (-not (Test-Path "$env:TWIN\tools\uv\uv.exe")) {
    New-Item -ItemType Directory -Force "$env:TWIN\tools\uv" | Out-Null
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip -OutFile "$env:TWIN\tools\uv.zip"
    Expand-Archive "$env:TWIN\tools\uv.zip" -DestinationPath "$env:TWIN\tools\uv" -Force
    Remove-Item "$env:TWIN\tools\uv.zip"
}

# Python 3.12 (Isaac Sim 6.x requires 3.12) managed by uv under E:\twin\python
uv python install 3.12
uv venv "$env:TWIN\isaacsim-env" --python 3.12
$py = "$env:TWIN\isaacsim-env\Scripts\python.exe"

# CUDA 12.8 PyTorch (Blackwell / RTX 50xx needs cu128+), then Isaac Sim with the extension caches
uv pip install --python $py torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
# the 5.5 GB isaacsim-extscache-kit wheel: resumable parallel download first (single streams kept dropping)
$whl = "$env:TWIN\cache\wheels\isaacsim_extscache_kit-6.2.0.0-cp312-none-win_amd64.whl"
New-Item -ItemType Directory -Force (Split-Path $whl) | Out-Null
& $py (Join-Path $here 'pdownload.py') "https://pypi.nvidia.com/isaacsim-extscache-kit/$(Split-Path -Leaf $whl)" $whl
if ((Get-FileHash $whl -Algorithm SHA256).Hash.ToLower() -ne 'bf26397ff036348b641a2b3644bc173282315b4676fa3d51f9c231730e5d8c81') { throw 'extscache-kit wheel hash mismatch' }
uv pip install --python $py $whl "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com `
    --index-strategy unsafe-best-match --prerelease=allow
# The Windows compatibility checker imports packaging; scene scripts use scipy/opencv/pyyaml
uv pip install --python $py packaging pyyaml scipy
Remove-Item $whl

# Separate env for evaluation / training (keeps Isaac Sim's pinned packages untouched)
uv venv "$env:TWIN\eval-env" --python 3.12
$epy = "$env:TWIN\eval-env\Scripts\python.exe"
uv pip install --python $epy torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $epy onnxruntime-gpu onnx ultralytics opencv-python-headless matplotlib

# Smoke test: headless RTX render of one frame
& $py (Join-Path $here 'scripts\smoke_render.py') --out "$env:TWIN\screens\smoke_render.png"
