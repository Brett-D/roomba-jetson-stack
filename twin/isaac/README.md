# Isaac Sim training twin (house + Roomba 550 + MYNT EYE D1000)

Phase 1 of the "both in parallel" plan: **Isaac Sim on the Windows PC for training** (synthetic data,
later learned behaviours), next to the 2D navigation twin on the Jetson. Everything here runs on the
PC; nothing is deployed to the robot.

```
twin/isaac/
  env.ps1, ros_env.ps1        cache/temp locations on E:\twin, bundled ROS 2 Jazzy for rclpy
  install_isaacsim.ps1        uv + Python 3.12 + CUDA PyTorch + isaacsim 6.2.0 (pip) + eval env
  pdownload.py                resumable parallel download (the 5.5 GB extscache wheel kept dropping)
  jetson/                     run on the Jetson: bag extraction, house mesh/grid export, IMU axis check
  robot/roomba550_d1000.urdf  Roomba 550 + D1000 (frames copied from the real /tf_static)
  scripts/                    Isaac Sim scripts (asset build, scene, ROS bridge, data generation)
  eval/                       detector baseline, sim-vs-real image statistics, fine-tune proof
```
Big files stay out of git, under `E:\twin` (install, caches, assets, datasets, runs, screenshots).

## 1. Install (Windows 11, RTX 5090, driver 621.14)

| | |
|---|---|
| Isaac Sim | **6.2.0** (Oct 2026, Kit 110.4.0), pip packages `isaacsim[all,extscache]==6.2.0.0` |
| Python | 3.12.16 (uv-managed, `E:\twin\python`); Isaac Sim 6.x requires 3.12 (3.11 was 5.x) |
| PyTorch | 2.11.0+cu128 (Blackwell needs CUDA 12.8+) |
| Renderer | RTX Real-Time 2.0 (path traced) on Vulkan; the Intel iGPU is skipped |
| Location | venv `E:\twin\isaacsim-env`, eval/train venv `E:\twin\eval-env`, caches `E:\twin\cache` |

Why pip and not the standalone zip: same 6.2.0 bits, but a normal venv (scripts run with plain
`python`), works with uv, and is the documented route for Isaac Lab 3.0 (GA, supports Isaac Sim 6.2,
pip install on Windows). Requirements met: Windows 11, RTX ≥ 4080 / 16 GB (5090, 32 GB),
driver ≥ 595.97, 32 GB+ RAM, SSD. WSL and the container are not supported for running the simulator
on Windows (the container is Linux-only).

```powershell
powershell -ExecutionPolicy Bypass -File twin\isaac\install_isaacsim.ps1
```
Installing means accepting the NVIDIA Omniverse License Agreement (`OMNI_KIT_ACCEPT_EULA=YES` in
`env.ps1`). Notes:
- Always dot-source `env.ps1` first: uv/pip/torch/warp/ultralytics caches and `%TEMP%` go to
  `E:\twin` (nothing large on C:, nothing on F:). Kit itself still keeps its small shader/log cache in
  `%LOCALAPPDATA%\ov`.
- `isaacsim-extscache-kit` is a single 5.5 GB wheel; over the VPN its download failed repeatedly
  ("peer closed connection"). `pdownload.py` fetches it in resumable parallel chunks; the install script checks its sha256
  (from `https://pypi.nvidia.com/isaacsim-extscache-kit/`) and installs it from the local file.
- Windows long paths were already enabled.
- Smoke test: `scripts\smoke_render.py` renders a frame headless (first start ~55 s, first frame
  ~100 s while shaders compile; later starts are faster).
- Render settings: use DLAA (or DLSS). With FXAA/TAA the real-time path tracer is left un-denoised
  (Immerkaer noise sigma 60 vs 1.4 with DLAA, `scripts\render_settings_test.py`).

## 2. Data from the robot (`jetson/`)

Work happens only in `/data/twin/isaac` on the Jetson (the source DB is read-only; `~/maps`,
`~/models` untouched).

```bash
python3 extract_bag.py /data/bags/robot_20261114_145937_fresh_map8 /data/twin/isaac/bag_extract
cp /data/twin/source/vslam.db /data/twin/isaac/vslam.db && bash export_house.sh   # grid, poses, cloud, mesh
python3 imu_axes.py
```
Measured from the fresh_map8 recording (2076 s, no MCAP index → one full scan, ~2.5 min):

| | real D1000 / Roomba |
|---|---|
| images | 640x360, **15.0 Hz** (not 30: output_scale 0.5 halves the rate too), left `bgr8`, mono `mono8`, depth `16UC1` mm registered to left |
| K (all three) | fx = fy = 349.2, cx = 318.035, cy = 174.522, no distortion (rectified) |
| right P[3] | -41.3376 → baseline **0.118378 m** (D1000-120) |
| FOV | 85.0° x 54.5° |
| IMU | 199.7 Hz, frame `camera_imu_frame` = left optical rotated 180° about x (x right, y up, z back); at rest accel (-0.14, 9.76, 0.09), noise σ ≈ 0.018 m/s², gyro σ ≈ 0.001 rad/s; no orientation |
| TF | base_link→camera_link (0.153, 0, 0.13); camera_link→left/right optical (0, ±0.05919, 0), rpy (-90°, 0, -90°) |
| drive | wheel base 0.235 m, wheel Ø 0.072 m, max wheel speed 0.4 m/s, wheel odom 30 Hz |

The camera lens is **0.13 m** above the floor (tape; depth reads ~0.11 m), so the data pipeline uses
0.13 m (jittered 0.11-0.16 m), not 0.24 m. The bag has no `/perception/person_detector/*`, no `/tf`
and no `/rtabmap/mapPath`; the robot path comes from the database's optimized graph instead
(`house_poses.txt`, 620 base_link poses in the map frame).

## 3. Assets

```powershell
. twin\isaac\env.ps1; cd twin\isaac\scripts
python build_house.py      # E:\twin\assets\usd\house.usd (+ house_textures\, house_stats.json)
python build_robot.py      # E:\twin\assets\usd\roomba550_d1000.usd (+ roomba_import\, _info.json)
python test_robot.py       # projection / depth / drive checks
python render_views.py     # top-down, oblique and sim-vs-real views -> E:\twin\screens
```
**House** (`build_house.py`), all in RTAB-Map's `map` frame so it lines up with the Jetson's 2D twin:
- `/House/Scan/*`: textured mesh from `rtabmap-export --mesh --texture` (Poisson, 6 × 4096² textures,
  97.7k faces), split by a geometric heuristic into wall / furniture labels; the bumpy scan floor
  and stretched gap-bridging triangles are dropped.
- `/House/Blocks/{wall,furniture}`: occupancy-grid cells extruded to the height of the scan points
  above them (≥ 1.3 m → full-height wall). These are the colliders.
- `/House/Floor` (collider) and `/House/Ceiling` at 2.5 m, variant set `look = scan | clean`
  (clean shows the blocks instead of the scan; used for domain randomisation).

**Robot** (`robot/roomba550_d1000.urdf` → Isaac Sim 6.2 URDF importer → `build_robot.py`):
articulation rooted at `base_link`, two velocity-driven wheel joints, frictionless front caster and
rear skid, tyre friction, D1000 housing, USD pinhole cameras on both optical frames reproducing the
real K (principal-point offset included), `IsaacImuSensor` at `camera_imu_frame`.
`test_robot.py`: a target projected 0.09 px from the real-K prediction, depth correct, 0.2 m/s for
3 s → 0.593 m (yaw drift 0.3°), 0.8 rad/s for 3 s → 123° (89 %, wheel scrub).

**People**: Isaac Sim People characters (10 outfits) animated by the Replicator Agent (IRA)
behaviour runtime on a NavMesh baked from the house (`twin_scene.py`). A bare UsdSkel binding does
not work: the characters use a Reallusion rig and the clips the Biped rig; IRA's motion library
retargets them.

## 4. Run the twin with ROS 2 (PC-local)

```powershell
. twin\isaac\ros_env.ps1; cd twin\isaac\scripts
python run_twin_ros.py --seconds 120 --people 3 --drive path --record E:\twin\runs\drive1
# second terminal
. twin\isaac\ros_env.ps1; python ros_check.py --seconds 25 [--drive]
```
Same topics / types / frame ids / encodings / rates as the real D1000 driver and Roomba driver:
`/camera/left/image_rect_color` (bgr8), `/camera/left/image_rect_mono`, `/camera/right/image_rect_mono`,
`/camera/depth/image_rect_raw` (16UC1 mm, 0 outside 0.3-7 m), the three `camera_info`, `/camera/imu/data_raw`
(200 Hz, real noise level), `/tf_static`, `/wheel/odom` (30 Hz), plus `/ground_truth/odom` (map →
base_link) and `/clock`. `/cmd_vel` (Twist) drives the wheels with the driver's kinematics (0.5 s
timeout, 0.4 m/s wheel cap); `--drive path` follows the recorded RTAB-Map path with pure pursuit.
ROS 2 is Isaac Sim's bundled Jazzy (`isaacsim.ros2.core`) used through rclpy in the sim process;
`ROS_DOMAIN_ID=69`, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, Fast DDS.
Verified (`E:\twin\eval\ros_check*.json`): images 15.0 Hz, IMU 200.0 Hz, odom 30.0 Hz, clock 60 Hz in
sim time; real-time factor 0.43 with 3 walking people; `/cmd_vel` moved the robot.

## 5. Synthetic data (`gen_dataset.py`)

```powershell
. twin\isaac\env.ps1; cd twin\isaac\scripts
python gen_dataset.py --frames 1600 --out E:\twin\datasets\v1
```
Robot-eye camera (real K, 640x360, 0.13 m ± jitter, pitch ±4°, roll ±2°) on drivable free space near
the recorded path, 75 % of the time aimed near a walking person. Randomised: lights (on/off,
intensity 0.25-2.5x, 2700-6500 K, dome), look (scan / clean) with floor / wall / furniture colours,
procedural railings and baluster rows (0.75-1.05 m, 9-16 cm spacing, 1.5-4.5 cm posts, metal / wood /
white), ArchVis furniture placement, people (8 characters walking / idling).
Output: `images/` RGB, `labels/` YOLO (0 person, 1 railing; boxes ≥ 30 % visible, ≥ 6 px tall),
`depth/` uint16 mm (D1000 validity), `semantic/` class ids (`classes.json`), `meta/` per-frame JSON,
`data.yaml`, `stats.json`, `README.md`.

## 6. Evaluate (`eval/`, use `E:\twin\eval-env`)

```powershell
python eval_yolo.py --synthetic E:\twin\datasets\v1 --real E:\twin\real_frames\left --out E:\twin\eval
python sim_vs_real.py --real E:\twin\real_frames\left --sim E:\twin\datasets\v1\images --out E:\twin\eval
python contact_sheet.py E:\twin\datasets\v1 E:\twin\screens\dataset_v1_sheet.jpg
python train_yolo.py --data E:\twin\datasets\v1 --epochs 5        # optional proof of the loop
```
`yolo_onnx.py` reproduces the Jetson node exactly (letterbox 384x640 grey 114, BGR→RGB, class 0,
conf 0.45, OpenCV NMS 0.5).

## Results (2026-11-15, see `E:\twin\eval`, `E:\twin\screens`)

| | |
|---|---|
| dataset v1 | 1600 frames in 607 s; 3136 person boxes in 1258 frames, 591 railing boxes in 455 frames; person distance median 2.4 m (10-90 %: 1.0-5.2 m) |
| baseline yolo11n (Jetson ONNX), all 1600 synthetic | precision 0.91, recall 0.22, AP50 0.32 at conf 0.45; recall 1 % for boxes < 40 px, 37 % for > 160 px; 33 % at < 1.5 m, 7 % beyond 5 m |
| same on the 160-frame val split | P 0.90, R 0.22, AP50 0.32 |
| fine-tuned 8 epochs on 1440 synthetic frames (2 classes) | val P 0.95, R 0.40, AP50 0.62 for person (ultralytics val: person mAP50 0.59, railing 0.22); exported 1x3x384x640 ONNX; not deployed |
| real fresh_map8 frames (1039, every 2 s) | no people were in this recording; baseline: 1 detection, a false positive on the stair balusters; fine-tuned: 0 |
| sim vs real (median) | brightness 50 vs 61, contrast 33 vs 24, Laplacian variance 175 vs 12 (real is much softer), crushed blacks 7 % vs 0 %, colour B:G:R 32:47:61 vs 37:68:58 (real is green); `eval/camera_model.py` brings them to 61 / 25 / 21 / 0 % / 38:62:65 |

## Known limits / next steps
- The scan mesh is noisy (IR depth from 0.13 m, Poisson surfaces with holes and shards); robot-eye
  views look like a low-poly cave. Good for geometry and labels, weak for photo-realism. Better:
  NuRec / 3D Gaussian splat of a dedicated video walk-through, or rebuild rooms with ArchVis assets
  laid out on the occupancy grid.
- Real-time factor 0.4 with two cameras at 15 Hz, depth and IMU at 200 Hz.
- Closed loop with the Jetson needs the network phase (Ethernet, firewall rule for DDS on the PC,
  same RMW/domain, or a Zenoh bridge).
