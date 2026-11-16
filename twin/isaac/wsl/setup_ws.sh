#!/usr/bin/env bash
# One-time: the closed-loop workspace in WSL (as root), separate from ~/roomba_ws and the other worktrees.
#   bash /mnt/e/twin/roomba-jetson-stack/twin/isaac/wsl/setup_ws.sh [repo] [ws]
set -euo pipefail
REPO=${1:-/mnt/e/twin/roomba-jetson-stack}
WS=${2:-/root/isaacloop_ws}
BRANCH=${BRANCH:-main}           # a branch of the Windows checkout ($REPO), e.g. main = origin/main there
if [ ! -d "$WS/.git" ]; then
  git clone -q --branch "$BRANCH" "$REPO" "$WS"
fi
cd "$WS"
git fetch -q --no-recurse-submodules origin "$BRANCH"
git checkout -q --detach "origin/$BRANCH"
# roomba-serial-ros2 (roomba_msgs, roomba_driver, roomba_driver_rs) at the pinned commit: from the Windows
# checkout's or ~/roomba_ws's submodule if either has it, else from GitHub
SUB=src/roomba-serial-ros2
PIN=$(git ls-tree HEAD "$SUB" | awk '{print $3}')
if [ "$(git -C "$SUB" rev-parse HEAD 2>/dev/null)" != "$PIN" ]; then
  URL=""
  for cand in "$REPO/$SUB" "/root/roomba_ws/$SUB"; do
    if git -C "$cand" cat-file -e "$PIN" 2>/dev/null; then URL=$cand; break; fi
  done
  git submodule init "$SUB"
  if [ -n "$URL" ] && [ -d "$SUB/.git" -o -f "$SUB/.git" ]; then
    git -C "$SUB" -c protocol.file.allow=always fetch -q "$URL" '+refs/heads/*:refs/remotes/local/*' \
      '+refs/remotes/origin/*:refs/remotes/local-origin/*' '+HEAD:refs/remotes/local/HEAD'
    git -C "$SUB" checkout -q --detach "$PIN"
  elif [ -n "$URL" ]; then
    git -c protocol.file.allow=always -c "submodule.$SUB.url=$URL" submodule update --force "$SUB"
  else
    git submodule update --force "$SUB"
  fi
fi
echo "workspace at $(git log --oneline -1), $SUB at $(git -C "$SUB" log --oneline -1)"
set +u
source /opt/ros/jazzy/setup.bash
source ~/.cargo/env
set -u
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-fuse-ld=mold"
colcon build --packages-up-to roomba_bringup roomba_rs roomba_web roomba_driver_rs --packages-ignore mynteye_d1000_driver basalt_vio_ros \
  --parallel-workers "${WORKERS:-8}" --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -25
