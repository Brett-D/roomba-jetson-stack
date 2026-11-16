#!/usr/bin/env bash
# TensorRT engines for the person detector, built in WSL (TensorRT 10.16 runtime from the WSL setup) from the
# ONNX copies on E:. trtexec comes from NVIDIA's libnvinfer-bin .deb, unpacked under $TOOLS (not installed).
#   bash build_engines.sh [home]     -> <home>/models/yolo11n_384x640_fp16.engine (+ yolo11s_synth if present)
set -euo pipefail
HOME_DIR=${1:-/root/isaacloop/home}
TOOLS=/root/isaacloop/tools
mkdir -p "$HOME_DIR/models" "$TOOLS"
TRTEXEC=$TOOLS/usr/src/tensorrt/bin/trtexec
if [ ! -x "$TRTEXEC" ]; then
  V=10.16.1.11-1+cuda13.2
  curl -sSLo "$TOOLS/libnvinfer-bin.deb" \
    "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/libnvinfer-bin_${V}_amd64.deb"
  dpkg-deb -x "$TOOLS/libnvinfer-bin.deb" "$TOOLS"
fi
build() {   # onnx engine
  [ -f "$2" ] && { echo "exists: $2"; return; }
  "$TRTEXEC" --onnx="$1" --saveEngine="$2" --fp16 --skipInference > "$2.log" 2>&1 && echo "built: $2" \
    || { echo "FAILED: $2 (see $2.log)"; tail -5 "$2.log"; }
}
build /mnt/e/twin/assets/yolo11n_384x640.onnx "$HOME_DIR/models/yolo11n_384x640_fp16.engine"
S=/mnt/e/twin/runs/train/yolo11s_synth_v2ae_neg/best_synth_384x640.onnx
[ -f "$S" ] && build "$S" "$HOME_DIR/models/yolo11s_synth_384x640_fp16.engine"
ls -la "$HOME_DIR/models"
