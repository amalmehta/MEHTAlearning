"""Horizon ablation: MEHTA's inner-loop horizon K_inner swept over 1x..16x the
short-horizon setting, everything else fixed.

MEHTA v2's inner loop runs the test-time post-training protocol, so K_inner is
both the inner-loop and the (meta-trained) post-training horizon. Each run is
mehta.experiments with --k-inner, one process and one CPU thread each, and
writes

  results/horizon/<env>/mehta_v2_k<K>_s<seed>.json    returns at every K, wall_sec, peak_rss_mb
  site/data/horizon.json                              mean ± std over seeds
  docs/img/horizon.png                                return vs horizon (± std bands), plus cost

  uv run python -m mehta.horizon_sweep               # 2 envs x 5 horizons x 3 seeds, then report
  uv run python -m mehta.horizon_sweep --jobs 6      # fewer parallel processes
  uv run python -m mehta.horizon_sweep --report-only

Existing result files are skipped unless --force, so an interrupted sweep resumes.
Wall-clock and peak memory are measured per process while --jobs runs share the
machine, so compare them within one sweep, not across machines.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .experiments import RESULTS, ROOT
from .train import METHODS

METHOD = "mehta_v2"                              # the current MEHTA config (v2/v3)
K_SHORT = METHODS["mehta_short"]["k_inner"]      # the short-horizon setting: 5
MULTS = [1, 2, 4, 8, 16]
HORIZONS = [K_SHORT * m for m in MULTS]          # 5, 10, 20, 40, 80 (default mehta_v2: 50)
SEEDS = [12, 13, 14]                             # fresh: seeds 0-11 are used by earlier rounds
ENVS = ["point_goal", "waypoints"]
OUT = RESULTS / "horizon"
SITE_DATA = ROOT / "site" / "data"
FIGS = ROOT / "docs" / "img"
SET_LABELS = {"id": "in-distribution goals", "ood_goal": "new goals", "ood_dyn": "shifted dynamics (60°)"}
SET_COLORS = {"id": "#9a3412", "ood_goal": "#c2410c", "ood_dyn": "#ea8a4f"}
ENV_LABELS = {"point_goal": "Point-Goal (dense return)", "waypoints": "Point-Waypoints (waypoints reached, of 3)"}


def result_path(env: str, k: int, seed: int) -> Path:
    return OUT / env / f"{METHOD}_k{k}_s{seed}.json"


def sweep(envs, horizons, seeds, jobs: int, force: bool):
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    todo = [(e, k, s) for s in seeds for e in envs for k in horizons if force or not result_path(e, k, s).exists()]
    print(f"{len(todo)} runs, {jobs} at a time")
    t0 = time.time()

    def one(job):
        e, k, s = job
        log = RESULTS / "logs" / f"horizon_{e}_k{k}_s{s}.log"
        with open(log, "w") as f:
            rc = subprocess.call([sys.executable, "-m", "mehta.experiments", "--env", e, "--method", METHOD,
                                  "--seed", str(s), "--k-inner", str(k), "--out", str(OUT)],
                                 stdout=f, stderr=subprocess.STDOUT)
        print(f"[{time.time() - t0:6.0f}s] {e} K_inner={k} seed {s}: {'ok' if rc == 0 else f'FAILED (see {log})'}",
              flush=True)
        return rc

    # longest runs first, so the tail of the sweep is not one slow job
    todo.sort(key=lambda j: -j[1])
    with ThreadPoolExecutor(jobs) as ex:
        failed = sum(rc != 0 for rc in ex.map(one, todo))
    if failed:
        sys.exit(f"{failed} runs failed")


def aggregate(envs, horizons, seeds) -> dict:
    """{env: {"return": {set: {k: [mean, std, n]}}, "wall_min" | "peak_mb" | "sample_grads_m": {k: ...}}};
    a seed's return = mean over that set's tasks at the last post-training checkpoint. sample_grads_m
    (meta-training sample-gradients, millions) is the deterministic compute; wall-clock and memory are measured."""
    ms = lambda xs: [round(float(np.mean(xs)), 4), round(float(np.std(xs)), 4), len(xs)]
    out, kfinal = {}, {}
    for e in envs:
        acc = {"return": {}, "wall_min": {}, "peak_mb": {}, "sample_grads_m": {}}
        for k in horizons:
            for s in seeds:
                p = result_path(e, k, s)
                if not p.exists():
                    continue
                run = json.loads(p.read_text())
                assert run["k_inner"] == run["train"]["k_inner"] == k, p
                for set_name, tasks in run["axis_a"][METHOD].items():
                    K = max(tasks[0], key=int)
                    kfinal[e] = int(K)
                    acc["return"].setdefault(set_name, {}).setdefault(k, []).append(
                        float(np.mean([t[K] for t in tasks])))
                acc["wall_min"].setdefault(k, []).append(run["wall_sec"] / 60)
                acc["peak_mb"].setdefault(k, []).append(run["peak_rss_mb"])
                acc["sample_grads_m"].setdefault(k, []).append(run["train"]["sample_grads"] / 1e6)
        out[e] = {"return": {sn: {str(k): ms(v) for k, v in d.items()} for sn, d in acc["return"].items()},
                  **{m: {str(k): ms(v) for k, v in acc[m].items()} for m in ("wall_min", "peak_mb", "sample_grads_m")}}
    return {"method": METHOD, "k_short": K_SHORT, "k_default": METHODS[METHOD]["k_inner"], "horizons": horizons,
            "seeds": list(seeds), "k_final": kfinal, "set_labels": SET_LABELS, "set_colors": SET_COLORS,
            "env_labels": ENV_LABELS, "curves": out}


def figure(agg: dict, dest: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fbf8f3", "axes.facecolor": "#fbf8f3"})
    envs = [e for e in ENV_LABELS if agg["curves"].get(e, {}).get("return")]
    fig, axes = plt.subplots(2, len(envs), figsize=(5.2 * len(envs), 6.8), squeeze=False)

    def band(ax, d, color, label, ls="-"):
        ks = sorted(d, key=int)
        x = np.array([int(k) for k in ks]); m = np.array([d[k][0] for k in ks]); sd = np.array([d[k][1] for k in ks])
        ax.plot(x, m, ls, marker="o", ms=4, color=color, label=label)
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.15, lw=0)

    def xaxis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(agg["horizons"], [f"{k}\n({k // agg['k_short']}×)" for k in agg["horizons"]])
        ax.minorticks_off()
        ax.axvline(agg["k_default"], color="#6b6157", lw=0.8, ls=":")
        ax.set_xlabel("inner-loop horizon K_inner")

    for ax, e in zip(axes[0], envs):
        for sn, d in agg["curves"][e]["return"].items():
            band(ax, d, SET_COLORS.get(sn), SET_LABELS.get(sn, sn))
        ax.set_title(ENV_LABELS[e], fontsize=10); xaxis(ax)
        ax.set_ylabel(f"return after K={agg['k_final'][e]} post-training steps")
        ax.text(agg["k_default"], 0.02, " default (50)", transform=ax.get_xaxis_transform(), fontsize=8, color="#6b6157")
        if len(agg["curves"][e]["return"]) > 1:
            ax.legend(frameon=False, fontsize=8)
    for ax, (m, title, ylab) in zip(axes[1], (("wall_min", "Measured wall-clock per run (shared machine)", "minutes"),
                                              ("peak_mb", "Measured peak memory per run", "MB (peak RSS)"))):
        for e, color in zip(envs, ("#1e3a8a", "#15803d")):
            band(ax, agg["curves"][e][m], color, ENV_LABELS[e].split(" (")[0])
        ax.set_title(title, fontsize=10); xaxis(ax); ax.set_ylabel(ylab); ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"MEHTA v2: final adaptation return vs. inner-loop horizon (mean ± std over {len(agg['seeds'])} seeds)",
                 fontsize=10)
    fig.tight_layout(); dest.parent.mkdir(parents=True, exist_ok=True); fig.savefig(dest, dpi=150); plt.close(fig)


def report(envs=ENVS, horizons=HORIZONS, seeds=SEEDS):
    agg = aggregate(envs, horizons, seeds)
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "horizon.json").write_text(json.dumps(agg))
    figure(agg, FIGS / "horizon.png")
    for e, c in agg["curves"].items():
        print(e)
        for sn, d in c["return"].items():
            print(f"  {sn:<15}" + "  ".join(f"K={k}: {d[k][0]:7.2f}±{d[k][1]:.2f}" for k in sorted(d, key=int)))
        for m in ("wall_min", "peak_mb", "sample_grads_m"):
            d = c[m]
            print(f"  {m:<15}" + "  ".join(f"K={k}: {d[k][0]:7.1f}±{d[k][1]:.1f}" for k in sorted(d, key=int)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=ENVS)
    ap.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        sweep(a.envs, a.horizons, a.seeds, a.jobs, a.force)
    report(a.envs, a.horizons, a.seeds)


if __name__ == "__main__":
    main()
