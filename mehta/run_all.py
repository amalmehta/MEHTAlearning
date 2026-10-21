"""One command regenerates every result, figure and demo file:

  uv run python -m mehta.run_all              # seeds 0 1 2, all envs and methods
  uv run python -m mehta.run_all --jobs 6     # fewer parallel processes
  uv run python -m mehta.run_all --version v2 --seeds 3 4 5 --methods pearl_cm pearl_dr_cm mehta_v2
  uv run python -m mehta.run_all --version v3 --seeds 6 7 8 --methods pearl_cm pearl_dr_cm mehta_v2 \
      --envs point_goal --sets id ood_goal ood_dyn <shift chosen by mehta.select_shift>
  uv run python -m mehta.run_all --version shift --seeds 9 10 11     # physics DR + shift curves

Runs are independent processes (one CPU thread each). Existing result files
are skipped unless --force, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .experiments import RESULTS

ENVS = ["point_goal", "waypoints"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=["pearl", "mehta", "mehta_short", "maml"],
                    help="v1: pearl mehta mehta_short maml; v2: pearl_cm pearl_dr_cm mehta_v2")
    ap.add_argument("--version", default="v1", choices=["v1", "v2", "v3", "shift"],
                    help="shift: train pearl_phys mehta_phys maml_phys (unless --methods), then mehta.shift_eval")
    ap.add_argument("--envs", nargs="+", default=ENVS)
    ap.add_argument("--sets", nargs="+", help="axis (a) evaluation sets, passed to every run")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.version == "shift":
        from .shift_eval import METHODS as SHIFT_METHODS
        if a.methods == ap.get_default("methods"):
            a.methods = SHIFT_METHODS
        a.sets = a.sets or ["id"]
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(e, m, s) for s in a.seeds for e in a.envs for m in a.methods
            if a.force or not (RESULTS / e / f"{m}_s{s}.json").exists()]
    print(f"{len(jobs)} runs, {a.jobs} at a time")
    t0 = time.time()

    def one(job):
        e, m, s = job
        log = RESULTS / "logs" / f"{e}_{m}_s{s}.log"
        with open(log, "w") as f:
            rc = subprocess.call([sys.executable, "-m", "mehta.experiments", "--env", e, "--method", m, "--seed", str(s),
                                  *(["--sets", *a.sets] if a.sets else [])],
                                 stdout=f, stderr=subprocess.STDOUT)
        print(f"[{time.time() - t0:6.0f}s] {e} {m} seed {s}: {'ok' if rc == 0 else f'FAILED (see {log})'}", flush=True)
        return rc

    with ThreadPoolExecutor(a.jobs) as ex:
        failed = sum(rc != 0 for rc in ex.map(one, jobs))
    if failed:
        sys.exit(f"{failed} runs failed")
    if a.version == "shift":
        subprocess.check_call([sys.executable, "-m", "mehta.shift_eval", "--envs", *a.envs, "--methods", *a.methods,
                               "--seeds", *map(str, a.seeds), "--jobs", str(a.jobs)])
        return
    subprocess.check_call([sys.executable, "-m", "mehta.report", "--version", a.version, "--seeds", *map(str, a.seeds)])
    if a.version == "v1" and (RESULTS / "models" / "point_goal_mehta_s0.pt").exists():
        subprocess.check_call([sys.executable, "-m", "mehta.export_demo"])


if __name__ == "__main__":
    main()
