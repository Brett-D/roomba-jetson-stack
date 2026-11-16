# How the learned parts were trained

This page is the overview of everything in this project that was **trained** (learned from data or
from simulated experience), how, with which code, and how well it works. It is written from the code,
configs and training logs; where a number could not be read from a log it was re-measured (and says
so).

**Status (2026-11-16): nothing trained in this project runs on the robot.**
- The robot's person detector uses an off-the-shelf, COCO-pretrained YOLO11n TensorRT engine
  (`~/models/yolo11n_384x640_fp16.engine`, loaded by `person_detector`); it was not trained here.
- The fine-tuned person/railing detector, the unstuck skill and the frontier ranker were trained and
  evaluated **in simulation only**. The two navigation skills are opt-in code paths that are off by
  default; the detector was never deployed. There is no real-robot validation of any of them yet.

| | 1. Person / railing detector | 2. Unstuck skill | 3. Frontier ranking |
|---|---|---|---|
| what it does | finds people (and railings) in the robot's camera image | drives the robot out of a stuck spot (nose to a wall, wedged under furniture, at a railing...) | picks which of the explorer's candidate frontier goals to drive to next |
| learning method | supervised fine-tuning of YOLO11 (Ultralytics) | reinforcement learning, PPO (RSL-RL) | cross-entropy method on a linear scorer, then PPO fine-tuning of a small network (own PyTorch code) |
| trained on | synthetic images rendered in Isaac Sim 6.2 (Replicator), plus real people-free frames as negatives | Isaac Lab 3.0 (PhysX): 4096 simulated Roombas in 1536 generated stuck situations | a batched 2D GPU simulator: 2000 generated houses |
| model | YOLO11s, 9.4 M parameters (YOLO11n 2.6 M) | MLP 596 -> 256 -> 128 -> 64 -> 2, 194 k parameters | CNN + MLP, 170 k parameters (146 k in the exported policy) |
| training budget (one RTX 5090) | 30 epochs, 27 min + 10 epochs, 10 min | 312 PPO iterations, 30.7 M simulated steps, 39 min | CEM 40 generations, 18 min + PPO 120 updates (450 k decisions), 17 min |
| headline result (simulation) | person recall 0.75 vs 0.28 for the COCO model at the same precision (0.96 vs 0.93); 0 false-positive frames on 312 held-out real frames | freed the robot in 86 % of 440 held-out generated situations vs 28 % for today's recovery; 85 % vs 30 % on 117 spots from the real floor plans | same final coverage with 10.6 % less time and 15.8 % less driving on 200 held-out houses |
| closed loop in `twin2d` (real stack, 2D twin) | - | called by the explorer when stuck: freed the robot in 92 of 93 calls (median 2.5 s); fewer bumper hits in the 10 scenarios, no coverage gain | no gain over the rule (coverage within noise, the time saving did not carry over) |
| deployed | no | no (`unstuck_skill` node, opt-in) | no (`goal_ranking: learned`, opt-in) |

Details of the two navigation skills: [learned-skills.md](learned-skills.md). Details of the Isaac
Sim twin and the detector data: [../twin/isaac/README.md](../twin/isaac/README.md).

## What is *not* learned

Most of what the robot does is hand-written logic, and some of the videos are replays. To keep them
apart from the trained parts:

- **The navigation stack is classical.** RTAB-Map / cuVSLAM mapping and localisation, Nav2 (planner,
  Regulated Pure Pursuit controller, behaviour tree with spin / back-up / wait recoveries, velocity
  smoother, collision monitor), `floor_obstacles`, keep-out marks (`ignore_zones`), `nav_coach`,
  `nav2_guard` and the **rule-based explorer** (frontier score = distance - 0.01 x frontier size, plus
  its look-around, reachability, retry and finish-at-home rules). The navigation fixes merged in
  November 2026 are changes to these rules, not learning.
- **Scripted path replay.** The Isaac Sim videos named `path_replay_*` replay a recorded robot path
  through the simulated house; the robot is not deciding anything. The `autonomous_*` Isaac Sim videos
  show the classical stack driving on its own (no learned parts).
- **Re-rendered rollouts.** `unstuck_classical_vs_learned.mp4` and `explore_rule_vs_learned.mp4`
  (`twin/learned/viz/`) are 2D drawings of logged simulation runs (Isaac Lab physics and the 2D
  exploration simulator); they are not robot footage.

Which simulator a number comes from matters:

| environment | what it is | used for |
|---|---|---|
| Isaac Sim 6.2 (Replicator) | photoreal renderer of the house remodelled from the floor plans | detector training data |
| Isaac Sim closed loop | Isaac Sim drives the robot's real ROS 2 stack in WSL | classical-stack testing (no learned skills yet) |
| Isaac Lab 3.0 (OVPhysX) | many small PhysX scenes in parallel, no rendering | unstuck skill training and evaluation |
| 2D exploration simulator (`twin/learned/explore`) | batched grid world in PyTorch on the GPU | frontier ranker training and evaluation |
| `twin2d` | 2D digital twin running the robot's real stack in real time | closed-loop checks of both skills |
| real robot | | none of the learned parts yet |

## 1. Person / railing detector (Isaac Sim synthetic data + YOLO fine-tune)

Purpose: the robot's person detector misses most people seen from 0.13 m above the floor (legs
only, close range, odd angles). Synthetic images from the robot's own viewpoint were rendered to
measure that and to fine-tune a better model; a second class, *railing*, was added because thin
balusters are what the depth camera misses.

**Data generation** (`twin/isaac/scripts/gen_dataset.py`, Isaac Sim 6.2.0 pip install,
`twin/isaac/install_isaacsim.ps1`): Omniverse Replicator annotators `rgb`,
`distance_to_image_plane`, `semantic_segmentation` and `bounding_box_2d_tight` on the house twin
rebuilt from the floor plans (`twin/isaac/remodel/`). Camera = the D1000's real left intrinsics
(640 x 360, fx = fy = 349.2), 0.13 m high (jittered 0.11-0.16 m), pitch +-4 deg, roll +-2 deg, placed
on drivable floor near the recorded robot path and aimed near a walking person 75 % of the time.
Domain randomisation per frame or every 25 frames: lights (on/off, 0.25-2.5x, 2700-6500 K, dome),
floor / wall / furniture colours, procedural railings and baluster rows (0.75-1.05 m high, 9-16 cm
spacing, 1.5-4.5 cm posts, metal / wood / white), furniture placement, 8 walking or idling people.
Labels: YOLO boxes for person (0) and railing (1), boxes at least 30 % visible and 6 px tall.
`eval/camera_model.py` applies a model of the real camera (softness, colour cast, noise,
auto-exposure) fitted by comparing image statistics with real frames (`eval/sim_vs_real.py`).

| dataset | frames | content |
|---|---|---|
| v1 | 1600 | scan-mesh house look, no camera model (first proof) |
| v2 | 11,958 (3 processes x 4 camera rigs, ~25 min) | floor-plan house, 26,344 person boxes in 78 % of frames, 37,492 railing boxes |
| v2-AE | same frames | v2 re-processed with the auto-exposure camera model |
| real negatives | 1039 real robot frames from one recording with nobody in it | first 70 % (727 frames, by time) used as background images, last 30 % (312) held out |

**Training** (`twin/isaac/eval/train_yolo.py`; Ultralytics 8.4.214, PyTorch 2.11.0+cu128, venv
`E:\twin\eval-env`): fine-tuning from the COCO-pretrained `yolo11n.pt` / `yolo11s.pt`, image size 640,
batch 64, 30 epochs, `optimizer=auto` (Ultralytics resolves it to AdamW with lr 0.001667 for runs under
10k iterations; the resumed yolo11n run's saved arguments show exactly that), lrf 0.01, 3 warm-up
epochs, weight decay 0.0005, seed 0, early-stop patience 10. Augmentation aimed at the sim-to-real gap:
hue 0.03, saturation 0.6, value 0.5, rotation 2 deg, translation 0.1, scale 0.4, horizontal flip 0.5,
mosaic 1.0. Validation split: 10 % in blocks of 100 consecutive frames per generator process (the four
cameras of one capture step never straddle train / val). Then a short **hard-negative round**: 10 more
epochs from the v2 weights on v2-AE plus the 727 real people-free frames as background images (0.5
warm-up epochs). Export: ONNX, 1 x 3 x 384 x 640 (the Jetson engine's input), opset 17.

| run | epochs | wall time (from `results.csv`) |
|---|---|---|
| yolo11n on v2 | 30 | 20 min |
| yolo11s on v2 | 30 | 27 min |
| yolo11n v2 -> v2-AE + real negatives | 10 | 8 min |
| yolo11s v2 -> v2-AE + real negatives | 10 | 10 min |

**Evaluation** (`twin/isaac/eval/eval_yolo.py`, the robot's exact pre- / post-processing reproduced by
`yolo_onnx.py`: letterbox 384 x 640, confidence 0.45, NMS IoU 0.5, person boxes >= 10 px). Person
detection on validation frames never trained on (v2 val: 1193 images, 2700 people), and false
positives on the 312 held-out real frames (no people in them, so every detection is false):

| model | v2 val P / R / AP50 | v2-AE val P / R / AP50 | real frames with a false person |
|---|---|---|---|
| COCO YOLO11n (what the robot runs) | 0.93 / 0.30 / 0.53 | 0.93 / 0.28 / 0.52 | 1 |
| YOLO11n fine-tuned on v2 | 0.96 / 0.74 / 0.86 | 0.96 / 0.66 / 0.81 | 19 |
| YOLO11s fine-tuned on v2 | 0.95 / 0.77 / 0.88 | 0.94 / 0.70 / 0.83 | 11 |
| YOLO11n v2 + real negatives | 0.94 / 0.73 / 0.85 | 0.95 / 0.73 / 0.85 | 0 |
| **YOLO11s v2 + real negatives** | 0.96 / 0.74 / 0.87 | 0.96 / 0.75 / 0.86 | **0** |

Best model, recall by box height (v2-AE val): 0 % under 40 px, 22 % 40-80 px, 59 % 80-160 px, 88 %
above 160 px (COCO model: 0 / 1 / 16 / 36 %). On the Jetson (TensorRT FP16, idle): YOLO11n 3.5 ms,
YOLO11s 5.4 ms per 384 x 640 frame. The fine-tuned models without real negatives fired on balusters and
dark door edges; the negatives round removed all of those on the held-out frames.

**Limitations / status.** Measured on synthetic images and on one real recording without people:
recall on *real* people is unknown (no labelled real frames with people yet). Railing detection was
trained (ultralytics val mAP50 0.22 for railing in the first run) but is not used by any node. Not
deployed: the robot still runs the COCO YOLO11n engine.

Reproduce: see [twin/isaac/README.md](../twin/isaac/README.md) sections 1, 5 and 6 (install, data
generation, training and evaluation commands); results in its "Results" section.

## 2. Unstuck skill (Isaac Lab, PPO)

Full reference: [learned-skills.md section 1](learned-skills.md#1-unstuck-skill).

**Algorithm**: PPO (clipped surrogate, GAE) from RSL-RL (`rsl-rl-lib` 5.5.1) through Isaac Lab's
`isaaclab_rl` wrapper. Configuration (`twin/learned/unstuck/train_unstuck.py`): 4096 parallel
environments, 24 steps per environment per iteration, 5 epochs x 4 mini-batches, learning rate 5e-4
with the adaptive KL schedule (target KL 0.01), clip 0.2, gamma 0.98, lambda 0.95, entropy coefficient
0.003, value-loss coefficient 1.0 (clipped), gradient-norm clip 1.0, initial action std 0.6,
observation normalisation, seed 7. Actor and critic: separate MLPs 256-128-64, ELU.

**Simulation** (`unstuck_env.py`, `scenarios.py`, `world.py`, `robot_usd.py`): Isaac Lab 3.0.0 in
kit-less mode on OVPhysX 0.6.4 (the PhysX solver as a library, no renderer), physics 50 Hz, control
10 Hz, 15 s episodes. A Roomba-sized robot (0.34 m, wheel base 0.235 m, real speed limits) starts in a
4 x 4 m tile holding one generated stuck situation: nose to a wall, corner, chair legs, narrow gap,
threshold, invisible railing (often with a stair drop behind), overhang at camera height, door leaf,
ring of bump keep-out marks, cliff edge, clutter. 1536 training tiles (seed 1); 440 held-out tiles
(seed 1001) and 117 tiles cut from the real house's two floor plans (`house_tiles.py`) are only used
for evaluation. The stack around the skill is emulated: velocity smoother limits, the collision
monitor's stop / slow-down zones (including floor_obstacles' "blind wall" when the camera sees nothing
closer than 0.3 m) and the driver's bump reflex.

**Observation** (596 values, mirrored in Rust by `src/roomba_rs/src/learned/unstuck.rs`): a 24 x 24
patch (0.1 m cells, robot frame) of lethal local-costmap cells built from a simulated depth ray fan
(0.3 m blind zone) plus bump keep-out marks; 6 light-bumper proximities; 2 bumper switches; wheel
encoder speed and turn rate; gyro yaw rate; previous action; motion over the last second; goal
distance and bearing; time used. **Action**: forward / turn velocity, scaled to 0.25 m/s forward,
0.15 m/s back, 0.8 rad/s.

**Reward**: +10 when the robot has 0.30 m of clearance to anything it can hit (ground truth, ends the
episode), +2 x progress toward the goal, +3 x clearance gained (up to 0.3 m), -1 per new bump, -0.1 per
step in contact, -0.5 per camera / lid contact, -10 for a drop or tip-over (ends the episode), -0.05 per
step, -0.02 x squared action change.

**Domain randomisation**: wheel gains 0.85-1.05 (each side), wheel torque limit 0.5-1.0, one control
step of latency (50 % of episodes), light-bumper gain 0.7-1.3 plus noise, 5 % depth-ray dropout, 1 %
costmap cell flips, the right bumper switch ignored (50 %, as on the real unit), gyro bias and noise,
start heading jitter.

**Budget** (`E:\twin\learned\runs\unstuck\main`, TensorBoard + console log): configured for 600
iterations, stopped by hand after 312 because the training success rate had been flat (mean 0.84 over
iterations 150-311); 312 x 4096 x 24 = 30.7 M environment steps (about 850 hours of simulated
driving) in 39 min on the RTX 5090 (median 19.9 k steps/s; the GPU was shared with other jobs).
Checkpoint `model_300.pt` was exported (`export_onnx.py`: normalisation + actor, 0.8 MB ONNX,
`src/roomba_rs/models/unstuck_policy.onnx`).

**Evaluation protocol** (`evaluate_unstuck.py`): one deterministic episode per held-out tile, no
randomisation, the right bumper ignored. Baseline = `baseline.py`, today's recovery emulated in the same
simulator on the same inputs: Nav2-like path following (rotate to heading, pure pursuit, progress
checker) and the behaviour tree's round robin (spin +90 deg, spin -180 deg, back up 0.10 m, clear the
costmap and wait 2 s), with the same smoother, collision monitor and bump reflex underneath.

| | 440 generated (classical / learned) | 117 from the real floor plans (classical / learned) |
|---|---|---|
| freed within 15 s | 0.28 / **0.86** | 0.30 / **0.85** |
| median time to get free | 6.9 s / **1.5 s** | 7.7 s / **1.1 s** |
| bumper hits per attempt | 0.29 / **0.17** | 0.15 / **0.01** |
| fell into a drop or tipped | 1.4 % / **0.2 %** | 2.6 % / **0 %** |

Weak spots: dense clutter (0.31 success) and chair legs (0.67).

**Status / limitations**: simulation only. The robot-side node (`unstuck_skill`, Rust, `tract` ONNX
inference, reproduces PyTorch within 5e-7) is built only with `-DLEARNED_SKILLS=ON` and is not started
by any launch file by default; with `unstuck_skill:=true` the Rust explorer calls it when it is stuck
(default off). In the 2D twin with the real stack (phase-2 A/B, 10 scenarios x 2 seeds + the real
upstairs floor plan x 4 seeds) the explorer called it 93 times, mostly after bumping an invisible
railing; it reported the robot free in 92 (median 2.5 s), the 10 scenarios had fewer bumper hits
(24 -> 15), and coverage did not improve
([learned-skills.md section 4](learned-skills.md#4-closed-loop-runs-in-the-2d-twin-wsl-real-stack)).
Sim-to-real gaps: rigid wheels, rugs not simulated as obstacles,
idealised costmap marks, a simplified light-bumper model whose real gains and floor offsets must be
calibrated on the robot (the node's `light_gains`, `light_floor_driving`, `light_floor_still`).

Reproduce (PowerShell, venv `E:\twin\isaaclab-env`, see learned-skills.md for the install):
```powershell
. twin\learned\env.ps1; cd twin\learned\unstuck
python robot_usd.py; python make_tiles.py; python house_tiles.py
python train_unstuck.py --num_envs 4096 --iters 600 --run main         # stopped after 312 iterations
python export_onnx.py E:\twin\learned\runs\unstuck\main\model_300.pt --out ..\..\..\src\roomba_rs\models\unstuck_policy.onnx
python evaluate_unstuck.py --controller classical --tiles data\tiles_eval.json
python evaluate_unstuck.py --controller policy --ckpt E:\twin\learned\runs\unstuck\main\model_300.pt --tiles data\tiles_eval.json
python evaluate_unstuck.py --controller policy --ckpt E:\twin\learned\runs\unstuck\main\model_300.pt --tiles data\tiles_house.json
```

## 3. Frontier ranking for the explorer (2D GPU simulator, CEM + PPO)

Full reference: [learned-skills.md section 2](learned-skills.md#2-exploration-learned-frontier-selection).

The explorer's rules still produce the candidate goals and every safety rule still applies; the
ranker only changes *which* candidate is driven to next.

**Simulator** (`twin/learned/explore/env2d.py`, `gpu.py`, `layouts.py`): a batched 2D grid world in
PyTorch (0.1 m cells, hundreds of houses at once, ~450 decisions/s) with Warp kernels for path tracing.
It reproduces the explorer's candidate rules and score from `explorer.rs`, simulates the depth camera
as an 85 deg, 3 m ray fan with a 0.3 m blind zone, plans Dijkstra paths on a Nav2-like costmap and
charges time and distance for driving. Training uses 2000 generated houses; doors open and close,
up to 2 people block paths, battery 30-100 %. Held out: 200 other generated houses and the real
house's two floor plans (never trained on).

**Inputs** (mirrored in Rust by `src/roomba_rs/src/learned/explore.rs`): a 64 x 64 robot-centred map
crop at 0.2 m (free / occupied / unknown), 10 global values (battery, time, heading, home offset,
failures...) and 16 features for each of up to 16 candidates (offset, straight and path distance,
frontier size, unknown area around it, turn, distance to home, failures / visits nearby, rule score,
rank...). **Output**: one score per candidate; the best one is driven to.

**Stage 1: cross-entropy method** (`train_cem.py`) on a linear scorer over the candidate features
(+ battery x home-distance terms), starting from the explorer's rule. Population 24, each member
explores the same 12 houses from the same start (common random numbers; the rule itself is member 0),
elite 25 %, initial sigma 0.5 (0.3 on the rule-score weight), mean / sigma smoothing 0.7, 40
generations, 18 min. Objective per episode: mean coverage over 30 min - 0.2 x (end time incl. the drive
home) / 30 min. (Population and house counts were confirmed by re-running generation 0, which
reproduces the logged numbers exactly.) Learned weights: path cost -0.33 next to the rule score -0.87,
unknown area around the frontier +0.24, distance from home +0.30, low battery x home distance -0.08.

**Stage 2: PPO fine-tuning** (`train_explore.py`, own implementation, Adam) of the full network
`logits = 30 x (theta . features) + net(map, globals, candidates)` with theta from stage 1 and the
network's output layer starting at zero: 256 houses x 16 decisions per update, 120 updates (450 k
decisions, 17 min), learning rate 1e-4, entropy 0.003, gamma 0.99, lambda 0.95, clip 0.2, 4 epochs,
value coefficient 0.5, gradient clip 1.0. Reward per decision: 10 x coverage gained - 0.01 x seconds
spent - 1 per failed goal. (Learning rate and entropy confirmed by re-running update 0, which reproduces
the logged losses.) PPO from scratch with the defaults (lr 3e-4, entropy 0.01) ran 81 updates / 304 k
decisions without improving on the rule. Network: 4 conv layers (16-32-32-32) + MLPs, 170 k
parameters with the value head. Exported to `src/roomba_rs/models/explore_ranker.onnx`
(`export_ranker.py`).

**Evaluation** (`evaluate_explore.py`, seed 100): every controller explores the same house from the same
start. 200 held-out houses (one start heading each) and each real floor plan from 16 start headings;
*static* (doors as drawn, no people, full battery) and *dynamic* (doors toggling, people, battery
30-100 %). Paired differences learned - rule with 95 % confidence intervals:

| set | coverage (rule -> learned) | time | distance | other |
|---|---|---|---|---|
| 200 held-out houses, static | 0.845 -> 0.837 (n.s.) | **-10.6 % +- 4.6** | **-15.8 % +- 5.6** | revisited share 25.2 % -> 21.0 %; median time to 90 % 405 -> 374 s |
| 200 held-out houses, dynamic | 0.672 -> 0.685 (n.s.) | -6.6 % +- 5.8 | -9.7 % +- 6.5 | failed goals 0.59 -> 0.42 per run |
| real upstairs plan, static (16) | 0.805 -> 0.797 | **-7.4 % +- 3.4** (faster in 13 of 16) | **-9.4 % +- 3.9** | |
| real downstairs plan, static (16) | 0.981 -> 0.980 | -6.9 % +- 5.3 | -8.9 % +- 7.1 | |
| real upstairs plan, dynamic (16) | 0.664 -> 0.693 | +3.0 % +- 12.3 | -0.8 % +- 13.6 | failed goals 0.31 -> 0.63 (worse) |
| real downstairs plan, dynamic (16) | 0.750 -> 0.723 | -7.3 % +- 17.2 | -8.6 % +- 18.8 | |

In short: about the same final coverage, reached with ~7-11 % less time and ~9-16 % less driving in
static houses; in the dynamic runs on the real plans (16 each) the differences are within noise and
the upstairs plan saw more failed goals.

**Status / limitations**: simulation only. In the Rust explorer, `goal_ranking: learned` (with
`goal_ranking_model`) re-ranks the surviving candidates; default `rule`, and the build feature is off by
default. The 2D simulator has no Nav2 dynamics (travel time from path length), simplified mapping and
visible railings. In `twin2d` (the real stack; phase-2 A/B on the current explorer, 10 scenarios x 2
seeds + the real upstairs floor plan x 4 seeds) the learned ranking did not beat the rule: about the
same coverage over the scenarios (62.8 vs 64.9 %, time -2 %), and on the floor-plan world slightly
more coverage (86.5 vs 84.3 %) with 16 % more driving; details in
[learned-skills.md section 4](learned-skills.md#4-closed-loop-runs-in-the-2d-twin-wsl-real-stack).
Keep `goal_ranking: rule`.

Reproduce:
```powershell
. twin\learned\env.ps1; cd twin\learned\explore
python make_layouts.py
python train_cem.py --pop 24 --houses 12 --gens 40 --run main
python train_explore.py --theta E:\twin\learned\runs\explore_cem\main\theta_mu.npy --ent 0.003 --lr 1e-4 --updates 120 --run ppo_from_cem
python export_ranker.py --ckpt E:\twin\learned\runs\explore\ppo_from_cem\ranker_final.pt
python evaluate_explore.py --theta E:\twin\learned\runs\explore_cem\main\theta_mu.npy --ckpt E:\twin\learned\runs\explore\ppo_from_cem\ranker_final.pt --out E:\twin\learned\runs\explore_eval_net
```

## Software and hardware

| | |
|---|---|
| GPU | NVIDIA RTX 5090 (32 GB), Windows 11; shared with other simulation jobs during training |
| detector | Isaac Sim 6.2.0 (pip, Python 3.12) for data; Ultralytics 8.4.214, PyTorch 2.11.0+cu128 for training (`E:\twin\eval-env`) |
| navigation skills | Isaac Lab 3.0.0 (kit-less, OVPhysX 0.6.4), RSL-RL 5.5.1, PyTorch 2.11.0+cu130, Warp 1.17 (`E:\twin\isaaclab-env`) |
| robot-side inference | Rust, `tract-onnx` 0.23 (pure Rust, CPU), cargo feature `learned` / CMake `-DLEARNED_SKILLS=ON` |

Large outputs (datasets, checkpoints, logs, videos) stay outside git under `E:\twin` on the training PC.
