"""One command regenerates every result, figure and demo file:

  uv run python -m mehta.run_all              # seeds 0 1 2, all envs and methods
  uv run python -m mehta.run_all --jobs 6     # fewer parallel processes

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

from .experiments import RESULTS, VARIANTS

ENVS = ["point_goal", "waypoints"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(e, m, s) for s in a.seeds for e in ENVS for m in VARIANTS
            if a.force or not (RESULTS / e / f"{m}_s{s}.json").exists()]
    print(f"{len(jobs)} runs, {a.jobs} at a time")
    t0 = time.time()

    def one(job):
        e, m, s = job
        log = RESULTS / "logs" / f"{e}_{m}_s{s}.log"
        with open(log, "w") as f:
            rc = subprocess.call([sys.executable, "-m", "mehta.experiments", "--env", e, "--method", m, "--seed", str(s)],
                                 stdout=f, stderr=subprocess.STDOUT)
        print(f"[{time.time() - t0:6.0f}s] {e} {m} seed {s}: {'ok' if rc == 0 else f'FAILED (see {log})'}", flush=True)
        return rc

    with ThreadPoolExecutor(a.jobs) as ex:
        failed = sum(rc != 0 for rc in ex.map(one, jobs))
    if failed:
        sys.exit(f"{failed} runs failed")
    subprocess.check_call([sys.executable, "-m", "mehta.report", "--seeds", *map(str, a.seeds)])
    if (RESULTS / "models" / "point_goal_mehta_s0.pt").exists():
        subprocess.check_call([sys.executable, "-m", "mehta.export_demo"])


if __name__ == "__main__":
    main()
