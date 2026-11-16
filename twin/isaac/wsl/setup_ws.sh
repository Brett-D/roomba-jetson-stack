#!/usr/bin/env bash
# One-time: the closed-loop workspace in WSL (as root), separate from ~/roomba_ws and the other worktrees.
#   bash /mnt/e/twin/roomba-jetson-stack/twin/isaac/wsl/setup_ws.sh [repo] [ws]
set -euo pipefail
REPO=${1:-/mnt/e/twin/roomba-jetson-stack}
WS=${2:-/root/isaacloop_ws}
BRANCH=${BRANCH:-twin-closedloop}
if [ ! -d "$WS/.git" ]; then
  git clone -q --branch "$BRANCH" "$REPO" "$WS"
fi
cd "$WS"
git fetch -q origin "$BRANCH" && git checkout -q "origin/$BRANCH" 2>/dev/null || true
# roomba-serial-ros2 (roomba_msgs, roomba_driver, roomba_driver_rs): take it from ~/roomba_ws if it has the pinned commit
SUB=src/roomba-serial-ros2
PIN=$(git ls-tree HEAD "$SUB" | awk '{print $3}')
if [ -z "$(ls -A "$SUB" 2>/dev/null)" ]; then
  if git -C "/root/roomba_ws/$SUB" cat-file -e "$PIN" 2>/dev/null; then
    git -c protocol.file.allow=always -c "submodule.$SUB.url=/root/roomba_ws/$SUB" submodule update --init "$SUB"
  else
    git submodule update --init "$SUB"
  fi
fi
set +u
source /opt/ros/jazzy/setup.bash
source ~/.cargo/env
set -u
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-fuse-ld=mold"
colcon build --packages-up-to roomba_bringup roomba_rs roomba_web --packages-ignore mynteye_d1000_driver basalt_vio_ros \
  --parallel-workers "${WORKERS:-8}" --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -25
