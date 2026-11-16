# Running the stack on the PC (WSL2)

The robot's ROS 2 stack also runs on the desktop PC (i7-13700K, 96 GB, RTX 5090, Windows 11) inside WSL2
Ubuntu 24.04, to run many 2D digital-twin navigation scenarios in parallel and, later, the perception stack
against the Isaac Sim twin (which runs natively on Windows). This page records the setup, what was verified
and the plan for the Isaac Sim connection. The helpers live in [`tools/pc_wsl/`](../tools/pc_wsl/).

## What is installed (WSL `Ubuntu-24.04`, user root)

| Part | Version | Source |
|---|---|---|
| ROS 2 Jazzy desktop, Nav2 (incl. STVL, keep-out filter), robot_localization, imu_tools, foxglove_bridge, rosbag2 + mcap, rtabmap_ros, colcon, rosdep | Nav2 1.3.13, STVL 2.5.5, RTAB-Map 0.23.13 | ROS apt (`ros2-apt-source`) |
| rmw_zenoh_cpp | 0.2.11 | ROS apt |
| CUDA toolkit (compiler, libraries, command-line tools; no Nsight GUIs except what Isaac ROS pulls in) | 13.2.2 | NVIDIA **WSL-Ubuntu** repo (no Linux driver in WSL: libcuda comes from Windows) |
| TensorRT runtime + headers (`libnvinfer10`, `libnvinfer-headers-dev`, plugin, ONNX parser) | 10.16.1.11+cuda13.2 | single `.deb`s from NVIDIA's ubuntu2404 repo (that repo is *not* an apt source here: it carries driver packages). `libnvinfer-dev` (3 GB of static libs) is skipped; `/usr/lib/x86_64-linux-gnu/libnvinfer.so` etc. are symlinks to the `.so.10`. Held with `apt-mark hold` |
| Isaac ROS: `isaac_ros_visual_slam_interfaces`, `isaac_ros_visual_slam`, `nvblox_ros` | 4.6.0 (as on the Jetson) | `isaac.download.nvidia.com/isaac-ros/release-4.6 noble main`, pinned to priority 100 so it never replaces ROS packages |
| VPI (Isaac ROS dependency) | 4.0.5 | `repo.download.nvidia.com/jetson/x86_64/noble r38.4`, pinned to 100 |
| Rust (rustup, stable), mold, libclang, OpenCV, libjpeg-turbo, libpng | 1.100.0, 2.30, 18, 4.6.0 | rustup / Ubuntu |

Disk: the distro grew by about 19 GB (ROS 3.3 GB, CUDA 4.8 GB, TensorRT ~1.5 GB, Isaac ROS + VPI ~6 GB,
workspace 1.3 GB).

### Setup steps (as root in WSL)

```bash
# ROS 2 Jazzy
apt install -y software-properties-common curl && add-apt-repository -y universe
V=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
curl -sSLo ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/$V/ros2-apt-source_$V.noble_all.deb" && dpkg -i ros2-apt-source.deb
apt update && apt install -y ros-jazzy-desktop ros-dev-tools ros-jazzy-navigation2 ros-jazzy-nav2-bringup \
  ros-jazzy-spatio-temporal-voxel-layer ros-jazzy-robot-localization ros-jazzy-imu-tools ros-jazzy-tf2-tools \
  ros-jazzy-foxglove-bridge ros-jazzy-rosbag2 ros-jazzy-rosbag2-storage-mcap ros-jazzy-rtabmap-ros ros-jazzy-rmw-zenoh-cpp \
  mold libclang-dev libjpeg-turbo8-dev libpng-dev libopencv-dev
# CUDA 13.2 from the WSL-Ubuntu repo (never install `cuda` or `cuda-drivers` in WSL)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb && dpkg -i cuda-keyring_1.1-1_all.deb
apt update && apt install -y cuda-compiler-13-2 cuda-libraries-dev-13-2 cuda-command-line-tools-13-2 cuda-nvml-dev-13-2
# TensorRT 10.16 runtime + headers as single debs, plus the .so dev links
B=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64; V=10.16.1.11-1+cuda13.2
for p in libnvinfer10 libnvinfer-headers-dev libnvinfer-plugin10 libnvonnxparsers10; do wget -c "$B/${p}_${V}_amd64.deb"; done
apt install -y ./libnvinfer*_${V}_amd64.deb ./libnvonnxparsers10_${V}_amd64.deb && apt-mark hold libnvinfer10 libnvinfer-headers-dev libnvinfer-plugin10 libnvonnxparsers10
for so in libnvinfer libnvinfer_plugin libnvonnxparser; do ln -sf $so.so.10 /usr/lib/x86_64-linux-gnu/$so.so; done
# Isaac ROS 4.6 apt repo (low priority) + VPI for x86_64
curl -sSL https://isaac.download.nvidia.com/isaac-ros/repos.key | gpg --dearmor -o /usr/share/keyrings/nvidia-isaac-ros.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.6 noble main" > /etc/apt/sources.list.d/nvidia-isaac-ros.list
printf 'Package: *\nPin: origin isaac.download.nvidia.com\nPin-Priority: 100\n' > /etc/apt/preferences.d/isaac-ros-low
curl -sSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | gpg --dearmor -o /usr/share/keyrings/nvidia-jetson-ota.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson-ota.gpg] https://repo.download.nvidia.com/jetson/x86_64/noble r38.4 main" > /etc/apt/sources.list.d/nvidia-vpi-x86.list
printf 'Package: *\nPin: origin repo.download.nvidia.com\nPin-Priority: 100\n' > /etc/apt/preferences.d/nvidia-vpi-x86-low
apt update && apt install -y --no-install-recommends ros-jazzy-isaac-ros-visual-slam-interfaces ros-jazzy-isaac-ros-visual-slam ros-jazzy-nvblox-ros
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && rustup update stable
```

## Build

```bash
git clone --recurse-submodules https://github.com/Brett-D/roomba-jetson-stack.git ~/roomba_ws && cd ~/roomba_ws
source /opt/ros/jazzy/setup.bash && rosdep update
rosdep install --from-paths src/roomba_rs src/roomba_bringup src/roomba_web src/roomba_perception src/roomba-serial-ros2 \
  --ignore-src -y --skip-keys "mynteye_d1000_driver basalt_vio_ros isaac_ros_visual_slam nvblox_ros nvblox_nav2 cuda-toolkit"
source tools/pc_wsl/roomba_env.sh        # ROS, cargo, CUDA_HOME, mold for x86_64, ROS_DOMAIN_ID 77 + localhost-only
colcon build --packages-up-to roomba_bringup roomba_rs roomba_web --packages-ignore mynteye_d1000_driver basalt_vio_ros \
  --parallel-workers 24 --cmake-args -DCMAKE_BUILD_TYPE=Release
```

- Skip the rosdep key `cuda-toolkit`: it maps to Ubuntu's old `nvidia-cuda-toolkit`.
- Use `--packages-ignore`, not `--packages-skip`, for the camera and Basalt packages: roomba_bringup's exec
  dependencies on them otherwise make colcon look for their install space.
- `roomba_env.sh` sets `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS=-C link-arg=-fuse-ld=mold`
  (`.cargo/config.toml` only selects mold for aarch64). A clean roomba_rs build takes about 50 s.

## Verified

- `cargo test --lib` in `src/roomba_rs`: 135 passed. `colcon test --packages-select roomba_rs roomba_driver`:
  40 tests, 0 failures (one earlier run had `roomba::tests::stream_frames_reach_callbacks_and_waiters` fail
  once under load: `deliver()` wakes `wait_for_frame` before the callbacks run, so the test races; 0/65
  failures when repeated).
- `ros2 pkg executables roomba_rs`: all 11 nodes, `person_detector` linked against TensorRT 10.16, CUDA 13.2
  and OpenCV 4.6.
- `tools/ab_driver.sh` (Python vs Rust driver on `tools/fake_roomba.py`): serial commands, shutdown bytes,
  services, `/roomba/mode`, odometry and logs identical. Only the 5 Hz `/roomba/sensors` sampled two extra
  short `oi_mode` flips for the Python driver (sampling phase).
- Smoke stack (`tools/pc_wsl/run_stack.sh`): the robot's Nav2 (`nav.launch.py` in robot mode with
  `nav2_robot.yaml`, Rust floor_obstacles, map_gap_filler, nav_coach, explorer), ignore_zones (keep-out
  filter), the Rust roomba_driver on `fake_roomba.py` (port from a generated params file) and `kinsim.py`
  (integrates `/cmd_vel` into `odom -> base_link`, publishes empty obstacle clouds) on a static 12 x 10 m
  map with a door. `smoke_check.py`: all lifecycle nodes active after 4 s, `/rtabmap/map_filled`,
  `/costmap_filter_info` and the global costmap present, `/wheel/odom` at 30 Hz, NavigateToPose through
  the door and back succeeded (~27-30 s per goal, 0.14 m end error).

### WSL clock steps and rclrs (fixed on this branch)

WSL2's clock on this PC runs ~0.3 % fast and its time sync steps `CLOCK_REALTIME` back by ~100 ms every
30 s (`tools/pc_wsl/clockwatch.py`). Jazzy's `rcl_wait` then returns `RCL_RET_TIMEOUT` for a wait that
ended for a ROS-time timer that is not due yet; rclrs 0.8 treats that as an error and stops spinning, so
the Rust nodes exited (`Error: Timeout occurred (RCL_RET_TIMEOUT)`). The roomba_driver died 25-100 s
after start in every run. `node_params::spin` / `spin_until` now spin again when only timeouts were
reported: three fixed drivers ran 200 s through 7 clock steps and still stop cleanly on SIGTERM. The same
can happen on the Jetson whenever NTP steps the clock back. (Sim-time runs are not affected: timers on
an overridden ROS clock don't set the wait timeout.)

## Parallel scenarios

`tools/pc_wsl/parallel.sh N [loops] [base_domain]` starts N isolated stacks (one `ROS_DOMAIN_ID` each,
`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`), drives NavigateToPose loops in all of them and samples CPU and
memory per stack (`measure.py`):

| Stacks | CPU while navigating (of 24 threads) | Memory used | Goals |
|---|---|---|---|
| 1 | 0.5 core (2 %) | ~0.3 GB (0.85 GB summed RSS) | 4/4 |
| 4 | 1.8 cores (8 %) | ~1.2 GB | 16/16 |
| 8 | 3.5 cores (15 %) | ~2.5 GB | 16/16 |
| 12 | 6-8 cores (27-34 %) | ~4 GB | 21/24, 23/24, 40/48: aborts from the robot's 20 ms BT `default_server_timeout` |
| 16, `SERVER_TIMEOUT_MS=200` | 10 cores (41 %) | ~6 GB | 60/64: one stack never came up (a lifecycle `change_state` response was dropped during the start-up discovery storm) |

Per stack the cost is Nav2 (bt_navigator, controller, planner, behaviours, smoother, collision monitor,
two lifecycle managers) plus ~5 % for the Rust nodes. CPU and RAM are not the limit; latency is.
Run **8 stacks comfortably** with the robot's parameters; for 12-16, use a sim params overlay with
`default_server_timeout: 200` (the harness does this with `SERVER_TIMEOUT_MS`), start stacks a few
seconds apart and restart any stack whose lifecycle nodes are not active after a minute. Use sim time
from the 2D twin where possible: slowdowns then don't fail goals. Keep `ROS_DOMAIN_ID` at or below 101:
higher IDs map into Linux's ephemeral port range (a run at 121-128 lost a service).
The 2D simulator's own CPU comes on top.

## Phase B: Isaac Sim (Windows) <-> ROS 2 in WSL

Measured with Isaac Sim 6.2's bundled Jazzy (`isaacsim.ros2.core`, which ships Fast DDS 2.14, Cyclone and
rmw_zenoh 0.2.9) from the twin's python on Windows and ROS nodes in WSL (default NAT networking, no
firewall or `.wslconfig` changes), using `tools/pc_wsl/dds_probe.py`, `img_probe.py` and `win_probe.ps1`:

| Transport | Result |
|---|---|
| Fast DDS, default multicast discovery | no discovery (multicast does not cross the WSL NAT) |
| Fast DDS, UDP-only profile with initial peers (Windows vEthernet IP 172.24.80.1 + WSL IP) | works only when the Windows participant starts first; with the WSL participant up first, the two exchanged discovery traffic for 25-45 s without ever matching (cause not found). Windows announces only its LAN/VPN/Default-Switch addresses, so WSL sends every packet to 4 addresses through the NAT |
| **Zenoh**: `rmw_zenohd` router in WSL, Windows as client of `tcp/127.0.0.1:7447` (WSL localhost forwarding) | works in either start order; 848x480 bgr8 and 1280x720 bgr8 at 30 Hz delivered completely (83 MB/s); via the WSL IP (`tcp/<wsl-ip>:7447`) 1280x720 at 60 Hz (165 MB/s) |

Isaac ROS 4.6 in WSL (NVIDIA only lists bare-metal x86_64 Ubuntu 24.04, but the apt packages run):
`tools/pc_wsl/gpu_test.sh` feeds `synth_stereo.py` (synthetic rectified stereo + depth with ground truth)
to cuVSLAM (composable node, `use_gpu: true`, cuVSLAM 15.0.0) and nvblox (`config/nvblox_robot.yaml`):
cuVSLAM tracked at 29 Hz with 0.8 cm median / 2.5 cm max error over 0.6 m of motion; nvblox published its
ESDF slice at 4 Hz with the walls in it. nvblox ships only sm_75 SASS + PTX, so the first start JIT-compiles
for Blackwell (~30 s; cached in `~/.nv/ComputeCache`, keep `CUDA_CACHE_MAXSIZE` large). Run cuVSLAM as a
composable node: `ros2 run isaac_ros_visual_slam isaac_ros_visual_slam` fails (Bazel runfiles).

### Recommended setup (nothing needs admin rights)

1. In WSL, run the router first and keep it running: `ros2 run rmw_zenoh_cpp rmw_zenohd` (a user systemd
   service later).
2. Start Isaac Sim with, in its environment: `RMW_IMPLEMENTATION=rmw_zenoh_cpp`,
   `ZENOH_CONFIG_OVERRIDE=mode="client";connect/endpoints=["tcp/127.0.0.1:7447"]`, and the twin's
   `ROS_DOMAIN_ID` (`run_twin_ros.py` only `setdefault`s Fast DDS, so these win; set them after
   dot-sourcing `ros_env.ps1`, which assigns Fast DDS; `ROS_AUTOMATIC_DISCOVERY_RANGE` does not apply to Zenoh).
3. In WSL, run the perception stack with `RMW_IMPLEMENTATION=rmw_zenoh_cpp` and the same domain ID:
   robot.launch.py with `camera:=false driver:=false use_sim_time:=true`, cuVSLAM + nvblox (pipeline B).
4. For more bandwidth (e.g. several 60 Hz image streams), connect Isaac Sim to `tcp/<wsl-ip>:7447`
   instead (the WSL IP changes when WSL restarts).

Approvals only for the fallbacks: Fast DDS across the NAT would need inbound UDP rules for Isaac Sim's
python/kit.exe (admin) and the start-order problem solved; `networkingMode=mirrored` in `.wslconfig` is
global to all WSL distros (Docker Desktop too) and needs `wsl --shutdown`, with UDP-only Fast DDS profiles
on both sides.
