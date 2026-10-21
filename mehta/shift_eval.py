"""Dynamics-shift evaluation: adaptation return against physics shift magnitude.

Loads full checkpoints written by mehta.experiments
(results/models/<env>_<method>_s<seed>_full.pt), post-trains each one on held-out
tasks at every shift magnitude in randomize.MAGNITUDES (s <= 1 inside the
training range, s > 1 outside), and writes

  results/shift/<env>/<method>_s<seed>.json    per-task returns at each K
  site/data/shift.json                         mean ± std over seeds
  docs/img/shift_curves.png                    the shift curves

  uv run python -m mehta.shift_eval                                   # evaluate + report
  uv run python -m mehta.shift_eval --methods pearl_phys mehta_phys maml_phys --seeds 9 10 11
  uv run python -m mehta.shift_eval --report-only

Same tasks, physics and adaptation RNG for every method: only the checkpoint changes.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .adapt import post_train
from .agent import Agent
from .envs import ENVS, make_task_sets
from .experiments import LAM, RESULTS, ROOT, VARIANTS
from .randomize import MAGNITUDES, PARAMS, Randomized, shift_signs, shifted

OUT = RESULTS / "shift"
SITE_DATA = ROOT / "site" / "data"
FIGS = ROOT / "docs" / "img"
CKPT = [0, 100, 400]
METHODS = ["pearl_phys", "mehta_phys", "maml_phys"]
SEEDS = [9, 10, 11]
LABELS = {"pearl_phys": "PEARL-DR alone", "pearl_phys_ft": "PEARL-DR + fine-tune",
          "mehta_phys": "MEHTA-DR", "maml_phys": "MAML-DR (first-order)"}
COLORS = {"pearl_phys": "#1e3a8a", "pearl_phys_ft": "#3b82f6", "mehta_phys": "#9a3412", "maml_phys": "#15803d"}
ENV_LABELS = {"point_goal": "Point-Goal (dense return)", "waypoints": "Point-Waypoints (waypoints reached, of 3)"}


def evaluate(env_name: str, method: str, seed: int, models: Path = RESULTS / "models", out_dir: Path = OUT) -> Path:
    torch.set_num_threads(1)
    t0 = time.time()
    path = models / f"{env_name}_{method}_s{seed}_full.pt"
    agent = Agent.from_checkpoint(torch.load(path, weights_only=False))
    Env = ENVS[env_name]
    tasks = make_task_sets(env_name, seed=seed).evals["id"]
    signs = shift_signs(len(tasks), seed)
    res = {"env": env_name, "method": method, "seed": seed, "checkpoint": path.name, "lam": LAM,
           "magnitudes": MAGNITUDES, "params": PARAMS, "variants": {}}
    for vname, (grad, lam) in VARIANTS[method].items():
        rows = res["variants"][vname] = {}
        for s in MAGNITUDES:
            # the adaptation RNG depends only on (seed, s), so every method sees the same draws
            rng = np.random.default_rng([30_000 + seed, int(s * 100)])
            rows[str(s)] = []
            for ti, (task, sg) in enumerate(zip(tasks, signs)):
                p = shifted(s, sg)
                E = Randomized(Env, seed=40_000 + 100 * seed + ti, physics=p)
                out, _, _ = post_train(agent, E, task, CKPT, lam, grad, rng)
                rows[str(s)].append({"physics": p.as_dict(), "in_range": p.in_training_range(),
                                     "return": {str(k): v for k, v in out.items()}})
        logging.getLogger("mehta").info("%s %s seed=%d %s done (%.0fs)", env_name, method, seed, vname, time.time() - t0)
    dest = out_dir / env_name / f"{method}_s{seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res))
    return dest


def aggregate(out_dir: Path, seeds) -> dict:
    """{env: {variant: {s: {K: [mean, std, n_seeds]}}}}; per-seed value = mean over tasks."""
    acc = {}
    for f in sorted(out_dir.glob("*/*_s*.json")):
        run = json.loads(f.read_text())
        if run["seed"] not in seeds:
            continue
        for v, rows in run["variants"].items():
            for s, tasks in rows.items():
                for K in tasks[0]["return"]:
                    acc.setdefault(run["env"], {}).setdefault(v, {}).setdefault(s, {}).setdefault(K, []).append(
                        float(np.mean([t["return"][K] for t in tasks])))
    ms = lambda xs: [round(float(np.mean(xs)), 4), round(float(np.std(xs)), 4), len(xs)]
    curves = {e: {v: {s: {K: ms(xs) for K, xs in d.items()} for s, d in vs.items()} for v, vs in ev.items()}
              for e, ev in acc.items()}
    return {"seeds": list(seeds), "magnitudes": MAGNITUDES, "train_edge": 1.0, "ckpt": CKPT, "params": PARAMS,
            "labels": LABELS, "colors": COLORS, "env_labels": ENV_LABELS, "curves": curves}


def figure(agg: dict, dest: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fbf8f3", "axes.facecolor": "#fbf8f3"})
    envs = [e for e in ENV_LABELS if e in agg["curves"]]
    fig, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 3.6), squeeze=False)
    kmax = str(max(agg["ckpt"]))
    for ax, e in zip(axes[0], envs):
        ax.axvspan(0, agg["train_edge"], color="#6b6157", alpha=0.08, lw=0)
        for v, d in agg["curves"][e].items():
            xs = sorted(d, key=float)
            x = [float(s) for s in xs]
            for K, ls, lab in (("0", ":", None), (kmax, "-", LABELS.get(v, v))):
                m = np.array([d[s][K][0] for s in xs]); sd = np.array([d[s][K][1] for s in xs])
                ax.plot(x, m, ls, marker="o", ms=3, color=COLORS.get(v), label=lab)
                if K == kmax:
                    ax.fill_between(x, m - sd, m + sd, color=COLORS.get(v), alpha=0.15, lw=0)
        ax.set_title(ENV_LABELS[e]); ax.set_xlabel("shift magnitude s (1 = edge of training range)")
        ax.text(0.5, 0.02, "training range", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color="#6b6157")
    axes[0][0].set_ylabel("return (higher is better)")
    axes[0][-1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Return vs. physics shift: solid = after K={kmax} post-training steps, dotted = K=0 "
                 f"(mean ± std over {len(agg['seeds'])} seeds)", fontsize=10)
    fig.tight_layout(); dest.parent.mkdir(parents=True, exist_ok=True); fig.savefig(dest, dpi=150); plt.close(fig)


def report(seeds, out_dir: Path = OUT):
    agg = aggregate(out_dir, seeds)
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "shift.json").write_text(json.dumps(agg))
    figure(agg, FIGS / "shift_curves.png")
    kmax = str(max(CKPT))
    for e, vs in agg["curves"].items():
        print(e)
        for v, d in vs.items():
            cells = "  ".join(f"s={s}: {d[s][kmax][0]:7.2f}±{d[s][kmax][1]:.2f}" for s in sorted(d, key=float))
            print(f"  {LABELS.get(v, v):<24} {cells}")


def _job(args):
    return evaluate(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=list(ENV_LABELS))
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if not a.report_only:
        jobs = [(e, m, s) for e in a.envs for m in a.methods for s in a.seeds
                if a.force or not (OUT / e / f"{m}_s{s}.json").exists()]
        missing = [j for j in jobs if not (RESULTS / "models" / f"{j[0]}_{j[1]}_s{j[2]}_full.pt").exists()]
        if missing:
            raise SystemExit(f"no checkpoint for {missing}; train first, e.g. "
                             f"uv run python -m mehta.run_all --version shift --seeds {' '.join(map(str, a.seeds))}")
        print(f"{len(jobs)} evaluations, {a.jobs} at a time")
        with ProcessPoolExecutor(a.jobs) as ex:
            for p in ex.map(_job, jobs):
                print("wrote", p.relative_to(ROOT), flush=True)
    report(a.seeds)


if __name__ == "__main__":
    main()
