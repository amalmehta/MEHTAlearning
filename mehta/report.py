"""Aggregate per-seed results into site/data/results.json and README figures.

  uv run python -m mehta.report [--results results] [--seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiments import RESULTS, ROOT

SITE_DATA = ROOT / "site" / "data"
FIGS = ROOT / "docs" / "img"

# names as a person would say them; hue families: MEHTA = terracotta, PEARL = blue, MAML = green
LABELS = {
    "mehta": "MEHTA", "mehta_short": "MEHTA, short inner loop", "mehta_lam0": "MEHTA, no proximal term",
    "pearl": "PEARL alone", "pearl_ft": "PEARL + naive fine-tune", "maml": "MAML (first-order)",
}
COLORS = {
    "mehta": "#c2410c", "mehta_short": "#f59e0b", "mehta_lam0": "#e8875a",
    "pearl": "#1d4ed8", "pearl_ft": "#60a5fa", "maml": "#15803d",
}
SET_LABELS = {"id": "In-distribution goals", "ood_goal": "Out-of-distribution goals",
              "ood_dyn": "Shifted dynamics (actions rotated 60°)"}


def load(results: Path, env: str, seeds):
    runs = []
    for f in sorted((results / env).glob("*_s*.json")):
        seed = int(f.stem.rsplit("_s", 1)[1])
        if seed in seeds:
            runs.append(json.loads(f.read_text()))
    return runs


def mean_std(xs):
    xs = np.asarray(xs, float)
    return [round(float(xs.mean()), 4), round(float(xs.std(ddof=0)), 4), len(xs)]


def axis_a(runs, metric=lambda r: r):
    """{set: {variant: {K: [mean, std, n_seeds]}}}; per-seed value = mean over tasks."""
    acc = {}
    for run in runs:
        for v, sets in run["axis_a"].items():
            for s, tasks in sets.items():
                for K in tasks[0]:
                    acc.setdefault(s, {}).setdefault(v, {}).setdefault(K, []).append(
                        float(np.mean([metric(t[K]) for t in tasks])))
    return {s: {v: {K: mean_std(xs) for K, xs in d.items()} for v, d in vs.items()} for s, vs in acc.items()}


def verdict(a, b):
    """The spec's rule: a beats b if mean gap > std_a + std_b."""
    gap = a[0] - b[0]
    if abs(gap) <= a[1] + b[1]:
        return "no clear difference"
    return "better" if gap > 0 else "worse"


def build(results: Path, seeds):
    out = {"seeds": list(seeds), "labels": LABELS, "colors": COLORS, "set_labels": SET_LABELS}
    pg, wp = load(results, "point_goal", seeds), load(results, "waypoints", seeds)
    out["n_runs"] = {"point_goal": len(pg), "waypoints": len(wp)}
    if pg:
        out["axis_a"] = axis_a(pg)
        out["axis_c"] = {}
        for run in pg:
            for v, d in run.get("axis_c", {}).items():
                e = out["axis_c"].setdefault(v, {"final": [], "forgetting": [], "R": []})
                e["final"].append(d["final_mean_return"]); e["forgetting"].append(d["forgetting"]); e["R"].append(d["R"])
        for v, e in out["axis_c"].items():
            e["final"], e["forgetting"] = mean_std(e["final"]), mean_std(e["forgetting"])
            e["R_seed0"] = e.pop("R")[0]
    if wp:
        out["axis_b"] = {"waypoints": axis_a(wp)["id"],
                         "success": axis_a(wp, metric=lambda r: float(r >= 3))["id"]}
    out["train_curves"] = {}
    for env, runs in (("point_goal", pg), ("waypoints", wp)):
        curves = {}
        for run in runs:
            for h in run["train"]["history"]:
                curves.setdefault(run["method"], {}).setdefault(h["iter"], []).append(h["val_return"])
        out["train_curves"][env] = {m: {it: mean_std(v) for it, v in sorted(d.items())} for m, d in curves.items()}
        out.setdefault("compute", {})[env] = {r["method"]: {"grad_steps": r["train"]["grad_steps"], "wall_sec": r["wall_sec"]} for r in runs if r["seed"] == min(seeds)}
    out["verdicts"] = verdicts(out)
    # recorded rollouts (seed 0): first 3 tasks, K = 0 and the last checkpoint
    out["traj"] = {}
    for env, runs in (("point_goal", pg), ("waypoints", wp)):
        for run in runs:
            if run["seed"] != 0:
                continue
            for v, sets in run["traj"].items():
                for s, tasks in sets.items():
                    out["traj"].setdefault(env, {}).setdefault(s, {})[v] = [
                        {"task": t["task"], "xy": {k: t["xy"][k] for k in (min(t["xy"], key=int), max(t["xy"], key=int))}}
                        for t in tasks[:3]]
    if pg:
        out["cfg"] = {k: pg[0]["train"][k] for k in ("train_cfg", "sac_cfg")} | {"lam": pg[0]["lam"]}
    return out


def verdicts(out):
    v = []
    a = out.get("axis_a", {})
    for s, d in a.items():
        if "mehta" not in d:
            continue
        kmax = max(d["mehta"], key=int)
        for b in ("pearl", "pearl_ft", "maml", "mehta_short"):
            if b in d:
                v.append({"axis": "a", "set": s, "K": kmax, "vs": b, "mehta": d["mehta"][kmax], "other": d[b][kmax],
                          "verdict": verdict(d["mehta"][kmax], d[b][kmax])})
    if "axis_b" in out:
        d = out["axis_b"]["waypoints"]
        kmax = max(d["mehta"], key=int) if "mehta" in d else None
        for b in ("pearl", "pearl_ft", "maml", "mehta_short"):
            if kmax and b in d:
                v.append({"axis": "b", "set": "waypoints", "K": kmax, "vs": b, "mehta": d["mehta"][kmax],
                          "other": d[b][kmax], "verdict": verdict(d["mehta"][kmax], d[b][kmax])})
    c = out.get("axis_c", {})
    if "mehta" in c:
        for b in ("pearl", "pearl_ft", "maml", "mehta_lam0"):
            if b in c:
                v.append({"axis": "c", "set": "final", "vs": b, "mehta": c["mehta"]["final"], "other": c[b]["final"],
                          "verdict": verdict(c["mehta"]["final"], c[b]["final"])})
    return v


def figures(out):
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fbf8f3", "axes.facecolor": "#fbf8f3"})
    if "axis_a" in out:
        sets = [s for s in ("id", "ood_goal", "ood_dyn") if s in out["axis_a"]]
        fig, axes = plt.subplots(1, len(sets), figsize=(4.2 * len(sets), 3.4), sharey=False)
        for ax, s in zip(np.atleast_1d(axes), sets):
            for v, d in out["axis_a"][s].items():
                Ks = sorted(d, key=int)
                x = [int(k) for k in Ks]
                m = np.array([d[k][0] for k in Ks]); sd = np.array([d[k][1] for k in Ks])
                ax.plot(x, m, "-o", ms=3, color=COLORS[v], label=LABELS[v])
                ax.fill_between(x, m - sd, m + sd, color=COLORS[v], alpha=0.15, lw=0)
            ax.set_title(SET_LABELS[s]); ax.set_xlabel("post-training gradient steps K")
        np.atleast_1d(axes)[0].set_ylabel("return (higher is better)")
        np.atleast_1d(axes)[-1].legend(frameon=False, fontsize=8)
        fig.suptitle(f"Point-Goal: return vs. post-training steps (mean ± std over {len(out['seeds'])} seeds)", fontsize=11)
        fig.tight_layout(); fig.savefig(FIGS / "adaptation.png", dpi=150); plt.close(fig)
    if "axis_b" in out:
        d = out["axis_b"]["waypoints"]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for v, dd in d.items():
            Ks = sorted(dd, key=int)
            m = np.array([dd[k][0] for k in Ks]); sd = np.array([dd[k][1] for k in Ks])
            ax.plot([int(k) for k in Ks], m, "-o", ms=3, color=COLORS[v], label=LABELS[v])
            ax.fill_between([int(k) for k in Ks], m - sd, m + sd, color=COLORS[v], alpha=0.15, lw=0)
        ax.set_ylim(0, 3.05); ax.set_xlabel("post-training gradient steps K (sparse reward only)")
        ax.set_ylabel("waypoints reached (of 3)"); ax.legend(frameon=False, fontsize=8)
        ax.set_title("Point-Waypoints: long sparse episodes")
        fig.tight_layout(); fig.savefig(FIGS / "waypoints.png", dpi=150); plt.close(fig)
    if out.get("axis_c"):
        c = out["axis_c"]
        names = [v for v in ("mehta", "mehta_lam0", "pearl_ft", "maml", "pearl") if v in c]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
        for ax, key, title in ((axes[0], "final", "final mean return (higher is better)"),
                               (axes[1], "forgetting", "forgetting (lower is better)")):
            ax.barh([LABELS[n] for n in names], [c[n][key][0] for n in names], xerr=[c[n][key][1] for n in names],
                    color=[COLORS[n] for n in names], alpha=0.85)
            ax.set_title(title, fontsize=10); ax.invert_yaxis()
        axes[1].set_yticklabels([])
        fig.suptitle("Continual post-training over 5 out-of-distribution tasks", fontsize=11)
        fig.tight_layout(); fig.savefig(FIGS / "continual.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    out = build(Path(a.results), a.seeds)
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "results.json").write_text(json.dumps(out))
    figures(out)
    for v in out["verdicts"]:
        print(f"axis {v['axis']} {v['set']:>9} MEHTA vs {LABELS[v['vs']]:<26} {v['mehta'][0]:8.3f}±{v['mehta'][1]:.3f} vs "
              f"{v['other'][0]:8.3f}±{v['other'][1]:.3f} -> {v['verdict']}")


if __name__ == "__main__":
    main()
