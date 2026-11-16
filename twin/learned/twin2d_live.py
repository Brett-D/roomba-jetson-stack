#!/usr/bin/env python3
"""Live, watchable 2D-twin demo of the learned unstuck skill (real stack, WSL / Jetson).

  source /root/learned_ws/learned_env.sh
  twin/learned/twin2d_live.py --ws /root/learned_ws --out /root/twin2d_data/runs/learned_live \
      --domain 40 --web_port 18099 --foxglove_port 18767 \
      --status /mnt/e/twin/learned_NOW.txt --password_file /mnt/e/twin/learned_watch_password.txt

Loops the scenarios (default nose_to_wall_live, wall_angle_live from twin/learned/twin2d/): each
run starts scripts/run_twin2d.sh with the web app and Foxglove on fixed ports (exit_on_finish, so
the launch ends when the scenario's duration is over), runs unstuck_skill next to it, calls the
skill once --skill_at s after the start, logs the result, then restarts with the next scenario.
The web app's HOME is in each run's out_dir; the login (auth.json) of the first run is copied
into every later run, so the password stays the same. Writes a status file after every phase.
Stop it with SIGTERM / SIGINT (twin/learned/twin2d_live_stop.sh): it stops the current run and
everything it started (all marked with LEARNED_LIVE=1 in their environment).
"""

import argparse
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = False


def on_signal(signum, frame):
    global STOP
    STOP = True


def killpg(p, sig=signal.SIGINT):
    if p is not None and p.poll() is None:
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            pass


def parse_result(text):
    if "Result:" not in text:
        return None
    tail = text.split("Result:")[-1]
    out = {}
    for key in ("outcome", "elapsed_s", "clearance_m", "bumps"):
        m = re.search(rf"\b{key}:\s*([-0-9.e]+)", tail)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"message:\s*'?([^'\n]*)", tail)
    if m:
        out["message"] = m.group(1).strip()
    m = re.findall(r"Goal finished with status:\s*(\w+)", text)
    out["status"] = m[-1] if m else "?"
    if "outcome" in out:
        out["status"] = {0: "SUCCEEDED", 1: "TIMED_OUT", 2: "FAILED", 3: "CANCELED"}.get(int(out["outcome"]), out["status"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="/root/learned_ws")
    ap.add_argument("--out", default="/root/twin2d_data/runs/learned_live")
    ap.add_argument("--domain", type=int, default=40)
    ap.add_argument("--web_port", type=int, default=18099, help="0 = no web app")
    ap.add_argument("--foxglove_port", type=int, default=18767, help="0 = no Foxglove bridge")
    ap.add_argument("--scenarios", nargs="+", default=["nose_to_wall_live", "wall_angle_live"])
    ap.add_argument("--loops", type=int, default=0, help="0 = until stopped")
    ap.add_argument("--skill_at", type=float, default=20.0,
                    help="call the skill by hand this long after the start; < 0: never (the explorer calls it, unstuck_skill:=true)")
    ap.add_argument("--world", default="/root/twin2d_data/world/house.yaml")
    ap.add_argument("--launch-args", default="", help="extra twin2d.launch.py arguments, e.g. 'unstuck_skill:=true goal_ranking:=learned'")
    ap.add_argument("--title", default="Learned unstuck skill - live 2D twin demo (WSL, real navigation stack + unstuck_skill)")
    ap.add_argument("--about", default="", help="status-file lines (separated by |) describing a run; default: the nose-to-wall demo")
    ap.add_argument("--status", default="")
    ap.add_argument("--password_file", default="")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "live.pid"), "w") as f:
        f.write(str(os.getpid()))
    models = os.path.join(a.ws, "install", "roomba_rs", "share", "roomba_rs", "models")
    setup = f"source /opt/ros/jazzy/setup.bash; source /root/roomba_ws/install/setup.bash; source {a.ws}/install/setup.bash"
    env = dict(os.environ, ROS_DOMAIN_ID=str(a.domain), ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST", LEARNED_LIVE="1",
               PYTHONUNBUFFERED="1", RCUTILS_LOGGING_BUFFERED_STREAM="1")
    webhome = os.path.join(a.out, "webhome", ".config", "roomba_web")
    started = datetime.datetime.now()
    hist_path = os.path.join(a.out, "history.json")
    history = []
    if os.path.exists(hist_path):
        try:
            history = json.load(open(hist_path))
        except ValueError:
            history = []
    state = {"loop": 0, "scenario": "", "phase": "starting", "since": started}
    about = a.about.split("|") if a.about else [
        f"Each run: robot starts stuck nose to the hallway wall; the learned skill is called ~{a.skill_at:.0f} s after the start;",
        "the explorer starts at 45 s and the run ends at 150 s, then the next run starts (scenarios alternate)."]
    extra = a.launch_args.split()

    def status():
        if not a.status:
            return
        now = datetime.datetime.now()
        lines = [
            a.title,
            f"Updated {now:%Y-%m-%d %H:%M:%S}; demo running since {started:%H:%M:%S}",
            "",
            f"Web app:  http://localhost:{a.web_port}   (password in {a.password_file.replace('/mnt/e/', 'E:/')})",
            f"Foxglove: ws://localhost:{a.foxglove_port}   (topics /twin/world, /twin/ground_truth, /local_costmap/costmap, /cmd_vel_nav, ...)",
            f"ROS_DOMAIN_ID {a.domain} (WSL, localhost only)",
            "",
            f"Now: run {state['loop']} ({state['scenario']}): {state['phase']} (since {state['since']:%H:%M:%S})",
            *about,
            "",
            "Stop: wsl -d Ubuntu-24.04 -u root -- bash /mnt/e/twin/learned/roomba-jetson-stack/twin/learned/twin2d_live_stop.sh",
            f"Loop PID (WSL): {os.getpid()}   logs: {a.out}",
            "",
            "Finished runs:",
        ]
        for h in history[-12:]:
            lines.append("  " + h)
        if not history:
            lines.append("  (none yet)")
        tmp = a.status + ".tmp"
        with open(tmp, "w", newline="\r\n") as f:
            f.write("\n".join(lines) + "\n")
        os.replace(tmp, a.status)

    def phase(text):
        state["phase"] = text
        state["since"] = datetime.datetime.now()
        print(f"[{state['since']:%H:%M:%S}] run {state['loop']} {state['scenario']}: {text}", flush=True)
        status()

    i = len(history)
    first = i
    while not STOP and (a.loops == 0 or i - first < a.loops):
        i += 1
        scen = a.scenarios[(i - 1) % len(a.scenarios)]
        state["loop"], state["scenario"] = i, scen
        out_dir = os.path.join(a.out, f"run{i:03d}_{scen}")
        shutil.rmtree(out_dir, ignore_errors=True)
        home_cfg = os.path.join(out_dir, "home", ".config", "roomba_web")
        if os.path.exists(os.path.join(webhome, "auth.json")):
            shutil.copytree(webhome, home_cfg, dirs_exist_ok=True)
        else:
            os.makedirs(out_dir, exist_ok=True)
        scen_path = os.path.join(HERE, "twin2d", scen + ".yaml")
        if not os.path.exists(scen_path):
            scen_path = os.path.join(a.ws, "src", "roomba_bringup", "config", "twin2d", scen + ".yaml")
        args = [os.path.join(a.ws, "scripts", "run_twin2d.sh"), f"scenario:={scen_path}", f"out_dir:={out_dir}", f"world:={a.world}",
                f"web:={'true' if a.web_port else 'false'}", f"foxglove:={'true' if a.foxglove_port else 'false'}", "exit_on_finish:=true"]
        if a.web_port:
            args.append(f"web_port:={a.web_port}")
        if a.foxglove_port:
            args.append(f"foxglove_port:={a.foxglove_port}")
        args += extra
        launch = subprocess.Popen(args, env=env, stdout=open(os.path.join(out_dir, "launch.log"), "w"), stderr=subprocess.STDOUT, start_new_session=True)
        call_log = os.path.join(out_dir, "unstuck_call.log")

        def start_skill(n):
            return subprocess.Popen(["bash", "-c", f"{setup}; exec ros2 run roomba_rs unstuck_skill --ros-args -p model:={models}/unstuck_policy.onnx"],
                                    env=env, stdout=open(os.path.join(out_dir, f"unstuck_skill{n or ''}.log"), "w"), stderr=subprocess.STDOUT,
                                    start_new_session=True)

        def start_call():
            cmd = f"{setup}; exec ros2 action send_goal --feedback /unstuck_skill roomba_skills_msgs/action/Unstuck '{{time_limit_s: 15.0}}'"
            return subprocess.Popen(["bash", "-c", cmd], env=env, stdout=open(call_log, "a"), stderr=subprocess.STDOUT, start_new_session=True)

        skill = start_skill(0) if a.skill_at >= 0 else None
        launch_log = os.path.join(out_dir, "launch.log")
        last_note = 0.0
        t0 = time.time()
        call = None
        res = None
        restarts = 0
        fb_n, fb_t = 0, 0.0
        phase("stack starting, robot stuck nose to the wall" if a.skill_at >= 0 else "stack starting")
        while not STOP and launch.poll() is None:
            time.sleep(0.5)
            if a.password_file and not os.path.exists(os.path.join(webhome, "auth.json")) and os.path.exists(os.path.join(home_cfg, "auth.json")):
                shutil.copytree(home_cfg, webhome, dirs_exist_ok=True)
                pw = os.path.join(home_cfg, "initial_password")
                if os.path.exists(pw):
                    with open(a.password_file, "w", newline="\r\n") as f:
                        f.write(open(pw).read().strip() + "\n")
            if a.skill_at < 0 and time.time() - last_note > 10:
                # explorer-driven runs: show what the explorer and the skill are doing
                last_note = time.time()
                try:
                    log = open(launch_log, errors="replace").read()
                except OSError:
                    log = ""
                calls, free = log.count("calling the unstuck skill"), log.count("Unstuck skill: free")
                said = re.findall(r"\[explorer\]: (.*)", log)
                note = f"exploring: {said[-1][:90]}" if said else "exploring"
                phase(f"{note}  (unstuck skill calls {calls}, freed {free})")
            if a.skill_at >= 0 and call is None and time.time() - t0 >= a.skill_at:
                call = start_call()
                call_t = time.time()
                fb_n, fb_t = 0, time.time()
                phase("learned unstuck skill driving (max 15 s)")
            if call is not None and res is None and call.poll() is not None:
                res = parse_result(open(call_log).read()) or {"status": "no result"}
                res["restarts"] = restarts
                phase(f"skill {res.get('status')} after {res.get('elapsed_s', float('nan')):.1f} s; waiting for the explorer (45 s), then exploring until 150 s")
            if call is not None and res is None and call.poll() is None:
                txt = open(call_log).read().split("### live demo")[-1]
                n = txt.count("Feedback:")
                if "Goal accepted" not in txt or n != fb_n:
                    fb_n, fb_t = n, time.time()
                elif time.time() - fb_t > 5.0 and restarts < 2:
                    # Safety net from before the clock-step fix (docs/learned-skills.md): if the
                    # node accepts a goal and then goes silent, restart it and call again.
                    restarts += 1
                    killpg(call, signal.SIGTERM)
                    killpg(skill, signal.SIGTERM)
                    time.sleep(1.0)
                    skill = start_skill(restarts)
                    time.sleep(4.0)
                    with open(call_log, "a") as f:
                        f.write(f"\n### live demo: goal accepted but no feedback for 5 s, unstuck_skill restarted ({restarts}), calling again\n")
                    call = start_call()
                    call_t = time.time()
                    fb_n, fb_t = 0, time.time()
                    phase(f"skill node stopped responding; restarted it ({restarts}) and called again")
                elif time.time() - fb_t > 40.0 or time.time() - call_t > 90.0:
                    killpg(call, signal.SIGTERM)
        for p in (call, skill):
            killpg(p, signal.SIGTERM)
        if STOP:
            killpg(launch, signal.SIGINT)
            try:
                launch.wait(timeout=25)
            except subprocess.TimeoutExpired:
                killpg(launch, signal.SIGKILL)
        r = {}
        try:
            r = json.load(open(os.path.join(out_dir, "result.json")))
        except (OSError, ValueError):
            pass
        cov = (r.get("coverage") or {}).get("final")
        bumps = len(r.get("bumps") or [])
        sk = "no call" if res is None else f"skill {res.get('status')} in {res.get('elapsed_s', float('nan')):.1f} s (clearance {res.get('clearance_m', float('nan')):.2f} m, bumps {int(res.get('bumps', 0))})"
        if restarts:
            sk += f" after {restarts} node restart(s)"
        run_txt = "no result.json" if not r else f"{r.get('outcome')} after {r.get('sim_time_s', 0):.0f} s, coverage {cov or 0:.3f}, {r.get('distance_m', 0):.1f} m, bumps {bumps}"
        if a.skill_at < 0:
            try:
                log = open(launch_log, errors="replace").read()
            except OSError:
                log = ""
            fin = ((r.get("explorer") or {}).get("final_message") or "")[:80]
            sk = f"unstuck skill calls {log.count('calling the unstuck skill')}, freed {log.count('Unstuck skill: free')}; {fin}"
        history.append(f"run {i:3d} {scen:<18s} {datetime.datetime.now():%H:%M}  {sk}; run: {run_txt}")
        json.dump(history, open(hist_path, "w"), indent=0)
        phase("run finished" if not STOP else "stopped")
        if not STOP and time.time() - t0 < 30:
            phase(f"run ended after {time.time() - t0:.0f} s (see {out_dir}/launch.log); retrying in 30 s")
            time.sleep(30)
        time.sleep(2)
    state["phase"] = "stopped"
    status()
    try:
        os.remove(os.path.join(a.out, "live.pid"))
    except OSError:
        pass


if __name__ == "__main__":
    sys.exit(main())
