# Environment for building and running the stack in WSL on the PC: . tools/pc_wsl/roomba_env.sh
_ws=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source /opt/ros/jazzy/setup.bash
[ -f "$_ws/install/setup.bash" ] && source "$_ws/install/setup.bash"
source ~/.cargo/env
export CUDA_HOME=/usr/local/cuda
case ":$PATH:" in *":/usr/local/cuda/bin:"*) ;; *) export PATH=/usr/local/cuda/bin:$PATH ;; esac
# roomba_rs/.cargo/config.toml selects mold only for aarch64 (the Jetson); GNU ld is very slow here too.
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-fuse-ld=mold"
# nvblox ships PTX for Blackwell: keep the JIT results (~30 s to rebuild) in a cache large enough
export CUDA_CACHE_MAXSIZE=${CUDA_CACHE_MAXSIZE:-4294967296}
# Keep this PC's ROS graph off the LAN (the robot is on it) unless a script overrides it.
export ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-77}
unset _ws
