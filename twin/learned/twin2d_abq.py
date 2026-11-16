#!/usr/bin/env python3
"""A/B of the learned skills in the 2D twin (real stack, WSL / Jetson): every scenario x seed x config
through tools/twin2d_run.py's run_job, round robin over the configs, N runs at a time (one ROS domain
each), then a summary table.

  source /root/learned_ws/learned_env.sh
  twin/learned/twin2d_abq.py --ws /root/learned_ws --out /root/learned_ab/phase2 --domains 30,31,32,33,34,35 \
      --seeds 1,2 --world /root/twin2d_data/world/house.yaml --plan-world /root/learned_ab/world_plan/plan.yaml
  twin/learned/twin2d_abq.py --out /root/learned_ab/phase2 --summary-only
  twin/learned/twin2d_abq.py --out /root/learned_ab/final,/root/learned_ab/phase2,/root/learned_ab/phase2b --summary-only

Configs (twin2d.launch.py arguments; the workspace must be built with -DLEARNED_SKILLS=ON):
  rule           the explorer as on main (goal_ranking rule, no unstuck skill)
  rule_skill     + unstuck_skill:=true (the explorer calls the skill when stuck)
  learned        goal_ranking:=learned
  learned_skill  both
Scenarios: every src/roomba_bringup/config/twin2d/*.yaml on --world, plus explore_from_dock on
--plan-world (rows "plan:explore_from_dock"). Web app off, no Foxglove (no ports in use).
Per run: coverage, simulated time, distance, bumps, stuck episodes and the longest, outcome, and the
explorer's unstuck calls (from launch.log).
"""

import argparse
import glob
import json
import os
import queue
import re
import sys
import threading
import time
import types

CONFIGS = {
    "rule": [],
    "rule_skill": ["unstuck_skill:=true"],
    "learned": ["goal_ranking:=learned"],
    "learned_skill": ["unstuck_skill:=true", "goal_ranking:=learned"],
}


def outcome(r):
    msg = ((r.get("explorer") or {}).get("final_message") or "")
    if "back at the start point" in msg:
        return "home"
    if "could not drive back" in msg:
        return "not home"
    return r.get("outcome", "?")


def run_row(cfg, scen, seed, d):
    try:
        r = json.load(open(os.path.join(d, "result.json")))
    except (OSError, ValueError):
        return None
    log = ""
    try:
        log = open(os.path.join(d, "launch.log"), errors="replace").read()
    except OSError:
        pass
    st = r.get("stuck") or {}
    eps = st.get("episodes") or []
    cov = r.get("coverage") or {}
    return {
        "config": cfg, "scenario": scen, "seed": seed,
        "coverage": cov.get("final"), "sim_s": r.get("sim_time_s"), "distance_m": r.get("distance_m"),
        "bumps": len(r.get("bumps") or []), "stuck": st.get("count", 0),
        "stuck_longest_s": max([e["to_s"] - e["from_s"] for e in eps], default=0.0),
        "idle_s": st.get("idle_total_s", 0.0), "outcome": outcome(r),
        "unstuck_calls": len(re.findall(r"calling the unstuck skill", log)),
        "unstuck_free": len(re.findall(r"Unstuck skill: free", log)),
        "unstuck_timeouts": len(re.findall(r"The unstuck skill did not finish within", log)),
        "watchdog_exits": len(re.findall(r"watchdog: no .* exiting to be respawned", log)),
    }


def summarise(out):
    """`out`: one output directory, or several comma-separated (a run in a later one replaces the
    same config / scenario / seed from an earlier one); the summary goes to the first."""
    found = {}
    for o in out.split(","):
        for cfg in CONFIGS:
            for root, prefix in ((os.path.join(o, cfg), ""), (os.path.join(o, cfg + "_plan"), "plan:")):
                for d in sorted(glob.glob(os.path.join(root, "*_s*"))):
                    m = re.match(r"(.+)_s(\d+)$", os.path.basename(d))
                    if m:
                        row = run_row(cfg, prefix + m.group(1), int(m.group(2)), d)
                        if row:
                            found[(cfg, row["scenario"], row["seed"])] = row
    rows = list(found.values())
    out = out.split(",")[0]
    json.dump(rows, open(os.path.join(out, "abq_rows.json"), "w"), indent=1)
    keys = ["scenario", "seed", "config", "coverage", "sim_s", "distance_m", "bumps", "stuck", "stuck_longest_s", "idle_s", "outcome", "unstuck_calls", "unstuck_free"]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in sorted(rows, key=lambda r: (r["scenario"], r["seed"], list(CONFIGS).index(r["config"]))):
        cells = []
        for k in keys:
            v = r[k]
            cells.append(f"{100 * v:.0f}%" if k == "coverage" and v is not None else (f"{v:.0f}" if isinstance(v, float) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| world | config | runs | coverage | sim s | distance m | bumps | stuck episodes | longest stuck s | home | unstuck calls (free / no answer) | watchdog exits |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for world, plan in (("scenarios", False), ("plan", True)):
        for cfg in CONFIGS:
            rs = [r for r in rows if r["config"] == cfg and r["coverage"] is not None and r["scenario"].startswith("plan:") == plan]
            if not rs:
                continue
            mean = lambda k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
            lines.append(f"| {world} | {cfg} | {len(rs)} | {100 * mean('coverage'):.1f}% | {mean('sim_s'):.0f} | {mean('distance_m'):.1f} | "
                         f"{sum(r['bumps'] for r in rs)} | {sum(r['stuck'] for r in rs)} | {max(r['stuck_longest_s'] for r in rs):.0f} | "
                         f"{sum(r['outcome'] == 'home' for r in rs)}/{len(rs)} | {sum(r['unstuck_calls'] for r in rs)} "
                         f"({sum(r['unstuck_free'] for r in rs)} / {sum(r.get('unstuck_timeouts', 0) for r in rs)}) | "
                         f"{sum(r.get('watchdog_exits', 0) for r in rs)} |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(out, "abq_summary.md"), "w").write(md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="/root/learned_ws")
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default="30,31,32,33,34,35")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--scenarios", default="", help="comma-separated names (default: all, plus the plan world)")
    ap.add_argument("--world", default="/root/twin2d_data/world/house.yaml")
    ap.add_argument("--plan-world", default="")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.summary_only:
        print(summarise(a.out))
        return
    sys.path.insert(0, os.path.join(a.ws, "tools"))
    import twin2d_run as TR  # noqa: E402

    scen_dir = os.path.join(a.ws, "src", "roomba_bringup", "config", "twin2d")
    names = a.scenarios.split(",") if a.scenarios else sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(scen_dir, "*.yaml")))
    items = []
    # the long plan-world runs first, so they don't trail at the end
    for name in (["plan:explore_from_dock"] if a.plan_world and not a.scenarios else []) + names:
        plan = name.startswith("plan:")
        stem = name.split(":", 1)[-1]
        for seed in [int(s) for s in a.seeds.split(",")]:
            for cfg in a.configs.split(","):
                world = (a.plan_world or a.world) if plan else a.world
                items.append((cfg, os.path.join(scen_dir, stem + ".yaml"), seed, os.path.join(a.out, cfg + ("_plan" if plan else "")),
                              ["web:=false", f"world:={world}"] + CONFIGS[cfg]))
    q = queue.Queue()
    for it in items:
        q.put(it)
    opts = types.SimpleNamespace(ws=a.ws, web_auth="", now_file="")
    t0 = time.time()
    done = [0]
    lock = threading.Lock()

    def worker(domain):
        while True:
            try:
                cfg, scen, seed, out_root, args = q.get_nowait()
            except queue.Empty:
                return
            os.makedirs(out_root, exist_ok=True)
            TR.run_job(scen, seed, out_root, args, 5.0, {"domain": domain}, opts)
            with lock:
                done[0] += 1
                print(f"[{time.time() - t0:5.0f}s] {done[0]}/{len(items)} done: {cfg} {os.path.basename(scen)} s{seed} (domain {domain})", flush=True)

    threads = [threading.Thread(target=worker, args=(int(d),)) for d in a.domains.split(",")]
    for t in threads:
        t.start()
        time.sleep(3)  # stagger the Nav2 bring-ups
    for t in threads:
        t.join()
    print(summarise(a.out))


if __name__ == "__main__":
    main()
