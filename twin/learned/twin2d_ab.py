#!/usr/bin/env python3
"""A/B closed-loop runs of the learned skills in the 2D twin (real stack, WSL / Jetson).

  twin/learned/twin2d_ab.py --ws /root/learned_ws --out /root/twin2d_data/runs/learned_ab --domain0 30 \
      explore_from_dock:1:rule explore_from_dock:1:learned nose_to_wall_skill:1:skill nose_to_wall_skill:1:none

Each run is <scenario>:<seed>:<mode>, started in parallel with its own ROS_DOMAIN_ID (domain0 + i),
no web app / Foxglove (no ports), through scripts/run_twin2d.sh of the overlay workspace --ws
(roomba_rs built with -DLEARNED_SKILLS=ON). Modes:
  rule     the explorer as configured (goal_ranking: rule)
  learned  explorer_params:={goal_ranking: learned, goal_ranking_model: <models>/explore_ranker.onnx}
  skill    also runs unstuck_skill and calls it once, --skill_at s after the start (the scenario's
           explorer autostart should be later), logging the action result
  none     like skill without the call (baseline for the same scenario file)
Scenario names are looked up in src/roomba_bringup/config/twin2d/ and twin/learned/twin2d/.
Writes <out>/ab_summary.json and prints a table.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def scenario_path(ws, name):
    for d in (os.path.join(ws, "src", "roomba_bringup", "config", "twin2d"), os.path.join(HERE, "twin2d")):
        p = os.path.join(d, name if name.endswith(".yaml") else name + ".yaml")
        if os.path.exists(p):
            return p
    sys.exit(f"no scenario {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ws", default="/root/learned_ws")
    ap.add_argument("--out", default="/root/twin2d_data/runs/learned_ab")
    ap.add_argument("--domain0", type=int, default=30)
    ap.add_argument("--skill_at", type=float, default=12.0)
    ap.add_argument("--world", default="/root/twin2d_data/world/house.yaml")
    ap.add_argument("--max_wall_s", type=float, default=2100.0)
    a = ap.parse_args()
    models = os.path.join(a.ws, "install", "roomba_rs", "share", "roomba_rs", "models")
    os.makedirs(a.out, exist_ok=True)
    procs = []
    for i, spec in enumerate(a.runs):
        name, seed, mode = spec.split(":")
        run = f"{name}_s{seed}_{mode}"
        out_dir = os.path.join(a.out, run)
        os.makedirs(out_dir, exist_ok=True)
        env = dict(os.environ, ROS_DOMAIN_ID=str(a.domain0 + i), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
        args = [os.path.join(a.ws, "scripts", "run_twin2d.sh"), f"scenario:={scenario_path(a.ws, name)}",
                f"out_dir:={out_dir}", f"seed:={seed}", f"world:={a.world}", "web:=false", "foxglove:=false"]
        if mode == "learned":
            args.append(f"explorer_params:={{goal_ranking: learned, goal_ranking_model: {models}/explore_ranker.onnx}}")
        log = open(os.path.join(out_dir, "launch.log"), "w")
        p = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        skill = None
        if mode == "skill":
            setup = f"source /opt/ros/jazzy/setup.bash; source /root/roomba_ws/install/setup.bash; source {a.ws}/install/setup.bash"
            skill = subprocess.Popen(["bash", "-c", f"{setup}; exec ros2 run roomba_rs unstuck_skill --ros-args -p model:={models}/unstuck_policy.onnx"],
                                     env=env, stdout=open(os.path.join(out_dir, "unstuck_skill.log"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
        procs.append({"run": run, "mode": mode, "p": p, "skill": skill, "env": env, "out": out_dir, "t0": time.time(), "called": False})
        print("started", run, "domain", a.domain0 + i, flush=True)
    while True:
        alive = 0
        for r in procs:
            if r["p"].poll() is None:
                alive += 1
                if r["mode"] == "skill" and not r["called"] and time.time() - r["t0"] >= a.skill_at:
                    r["called"] = True
                    setup = f"source /opt/ros/jazzy/setup.bash; source /root/roomba_ws/install/setup.bash; source {a.ws}/install/setup.bash"
                    cmd = f"{setup}; ros2 action send_goal --feedback /unstuck_skill roomba_skills_msgs/action/Unstuck '{{time_limit_s: 15.0}}'"
                    r["call"] = subprocess.Popen(["bash", "-c", cmd], env=r["env"], stdout=open(os.path.join(r["out"], "unstuck_call.log"), "w"),
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                if time.time() - r["t0"] > a.max_wall_s:
                    os.killpg(r["p"].pid, signal.SIGINT)
        if alive == 0:
            break
        time.sleep(2)
    rows = []
    for r in procs:
        for key in ("skill", "call"):
            q = r.get(key)
            if q is not None and q.poll() is None:
                try:
                    os.killpg(q.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        res = {}
        try:
            res = json.load(open(os.path.join(r["out"], "result.json")))
        except OSError:
            pass
        cov = res.get("coverage", {}) or {}
        st = (res.get("explorer") or {}).get("status", {}) or {}
        row = {
            "run": r["run"], "outcome": res.get("outcome"), "sim_time_s": res.get("sim_time_s"),
            "coverage": cov.get("final"), "reachable": cov.get("reachable_final"),
            "t50": cov.get("t_50pct_s"), "t80": cov.get("t_80pct_s"), "t90": cov.get("t_90pct_s"),
            "distance_m": res.get("distance_m"), "bumps": len(res.get("bumps") or []),
            "stuck": (res.get("stuck") or {}).get("count"), "stuck_s": (res.get("stuck") or {}).get("total_s"),
            "reached": st.get("reached"), "failed": st.get("failed"),
        }
        cl = os.path.join(r["out"], "unstuck_call.log")
        if os.path.exists(cl):
            txt = open(cl).read()
            row["unstuck_call"] = txt[-600:]
        rows.append(row)
    json.dump(rows, open(os.path.join(a.out, "ab_summary.json"), "w"), indent=1)
    keys = ["run", "outcome", "sim_time_s", "coverage", "reachable", "t50", "t80", "t90", "distance_m", "bumps", "stuck", "reached", "failed"]
    print(" | ".join(keys))
    for row in rows:
        print(" | ".join(str(row.get(k)) for k in keys))


if __name__ == "__main__":
    main()
