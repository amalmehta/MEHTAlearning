"""Aggregate per-seed results into site/data/results.json and README figures.

  uv run python -m mehta.report [--version v1|v2] [--results results] [--seeds ...]
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
    "mehta_v2": "MEHTA v2", "mehta_v2_lam0": "MEHTA v2, no proximal term",
    "pearl_cm": "PEARL alone (compute-matched)", "pearl_cm_ft": "PEARL + fine-tune (compute-matched)",
    "pearl_dr_cm": "PEARL-DR alone", "pearl_dr_cm_ft": "PEARL-DR + fine-tune",
}
COLORS = {
    "mehta": "#c2410c", "mehta_short": "#f59e0b", "mehta_lam0": "#e8875a",
    "pearl": "#1d4ed8", "pearl_ft": "#60a5fa", "maml": "#15803d",
    "mehta_v2": "#9a3412", "mehta_v2_lam0": "#fb923c",
    "pearl_cm": "#1e3a8a", "pearl_cm_ft": "#3b82f6", "pearl_dr_cm": "#0e7490", "pearl_dr_cm_ft": "#22d3ee",
}
# v1 = the published pilot (seeds 0-2); v2 = the revised method on fresh seeds 3-5
VERSIONS = {
    "v1": dict(methods={"pearl", "mehta", "mehta_short", "maml"}, seeds=[0, 1, 2], hero="mehta",
               base_a=("pearl", "pearl_ft", "maml", "mehta_short"), base_c=("pearl", "pearl_ft", "maml", "mehta_lam0"),
               cont=("mehta", "mehta_lam0", "pearl_ft", "maml", "pearl"), out="results.json", suffix=""),
    "v2": dict(methods={"pearl_cm", "pearl_dr_cm", "mehta_v2"}, seeds=[3, 4, 5], hero="mehta_v2",
               base_a=("pearl_cm", "pearl_cm_ft", "pearl_dr_cm", "pearl_dr_cm_ft"),
               base_c=("pearl_cm_ft", "pearl_dr_cm_ft", "mehta_v2_lam0"),
               cont=("mehta_v2", "mehta_v2_lam0", "pearl_cm_ft", "pearl_dr_cm_ft"), out="results_v2.json", suffix="_v2"),
    # v3 = same methods as v2 on fresh seeds 6-8, plus the held-out shift picked by select_shift
    "v3": dict(methods={"pearl_cm", "pearl_dr_cm", "mehta_v2"}, seeds=[6, 7, 8], hero="mehta_v2",
               base_a=("pearl_cm", "pearl_cm_ft", "pearl_dr_cm", "pearl_dr_cm_ft"),
               base_c=("pearl_cm_ft", "pearl_dr_cm_ft", "mehta_v2_lam0"),
               cont=("mehta_v2", "mehta_v2_lam0", "pearl_cm_ft", "pearl_dr_cm_ft"), out="results_v3.json", suffix="_v3"),
}
SET_LABELS = {"id": "In-distribution goals", "ood_goal": "Out-of-distribution goals",
              "ood_dyn": "Shifted dynamics (actions rotated 60°)",
              "shift_rot120": "Actions rotated 120°", "shift_invert": "Actions inverted (180°)",
              "shift_mirror": "Actions mirrored left-right", "shift_wind": "Strong wind (drift 0.06/step)",
              "shift_ood_rot60": "New goals + actions rotated 60°"}


def load(results: Path, env: str, seeds, methods):
    runs = []
    for f in sorted((results / env).glob("*_s*.json")):
        method, seed = f.stem.rsplit("_s", 1)
        if int(seed) in seeds and method in methods:
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


def build(results: Path, seeds, ver="v1"):
    V = VERSIONS[ver]
    out = {"version": ver, "seeds": list(seeds), "labels": LABELS, "colors": COLORS, "set_labels": SET_LABELS, "hero": V["hero"]}
    pg, wp = load(results, "point_goal", seeds, V["methods"]), load(results, "waypoints", seeds, V["methods"])
    out["n_runs"] = {"point_goal": len(pg), "waypoints": len(wp)}
    if pg:
        out["axis_a"] = axis_a(pg)
        # oracle reference (an estimate: greedy controller that knows goal and dynamics) and gap closure
        from .envs import make_task_sets, oracle_return
        out["oracle"] = {s: round(float(np.mean([np.mean([oracle_return(t) for t in make_task_sets("point_goal", seed=sd).evals[s]])
                                                  for sd in seeds])), 3) for s in out["axis_a"]}
        out["closure"] = {}
        for s, d in out["axis_a"].items():
            for v, dd in d.items():
                k0, kmax = "0", max(dd, key=int)
                gap0 = out["oracle"][s] - dd[k0][0]
                if gap0 > 1e-9:
                    out["closure"].setdefault(s, {})[v] = round((dd[kmax][0] - dd[k0][0]) / gap0, 3)
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
        out.setdefault("compute", {})[env] = {r["method"]: {"grad_steps": r["train"]["grad_steps"], "sample_grads": r["train"].get("sample_grads"),
                                                            "wall_sec": r["wall_sec"]} for r in runs if r["seed"] == min(seeds)}
    out["verdicts"] = verdicts(out, V)
    # recorded rollouts (seed 0): first 3 tasks, K = 0 and the last checkpoint
    out["traj"] = {}
    for env, runs in (("point_goal", pg), ("waypoints", wp)):
        for run in runs:
            if run["seed"] != min(seeds):
                continue
            for v, sets in run["traj"].items():
                for s, tasks in sets.items():
                    out["traj"].setdefault(env, {}).setdefault(s, {})[v] = [
                        {"task": t["task"], "xy": {k: t["xy"][k] for k in (min(t["xy"], key=int), max(t["xy"], key=int))}}
                        for t in tasks[:3]]
    if pg:
        out["cfg"] = {k: pg[0]["train"][k] for k in ("train_cfg", "sac_cfg")} | {"lam": pg[0]["lam"]}
    return out


def verdicts(out, V):
    """Hero method vs each baseline at the largest K. The key "mehta" holds the
    hero's numbers (MEHTA in v1, MEHTA v2 in v2), so the site renders both."""
    h, v = V["hero"], []
    a = out.get("axis_a", {})
    for s, d in a.items():
        if h not in d:
            continue
        kmax = max(d[h], key=int)
        for b in V["base_a"]:
            if b in d:
                v.append({"axis": "a", "set": s, "K": kmax, "vs": b, "mehta": d[h][kmax], "other": d[b][kmax],
                          "verdict": verdict(d[h][kmax], d[b][kmax])})
    if "axis_b" in out:
        for metric in ("waypoints", "success"):
            d = out["axis_b"][metric]
            kmax = max(d[h], key=int) if h in d else None
            for b in V["base_a"]:
                if kmax and b in d:
                    v.append({"axis": "b", "set": "waypoints" if metric == "waypoints" else "waypoints_success", "K": kmax, "vs": b,
                              "mehta": d[h][kmax], "other": d[b][kmax], "verdict": verdict(d[h][kmax], d[b][kmax])})
    c = out.get("axis_c", {})
    if h in c:
        for b in V["base_c"]:
            if b in c:
                v.append({"axis": "c", "set": "final", "vs": b, "mehta": c[h]["final"], "other": c[b]["final"],
                          "verdict": verdict(c[h]["final"], c[b]["final"])})
                # forgetting: lower is better, so compare negated values
                neg = lambda x: [-x[0], x[1], x[2]]
                v.append({"axis": "c", "set": "forgetting", "vs": b, "mehta": c[h]["forgetting"], "other": c[b]["forgetting"],
                          "verdict": verdict(neg(c[h]["forgetting"]), neg(c[b]["forgetting"]))})
    return v


def figures(out, V):
    sfx = V["suffix"]
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fbf8f3", "axes.facecolor": "#fbf8f3"})
    if "axis_a" in out:
        sets = [s for s in SET_LABELS if s in out["axis_a"]]
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
        for ax, s in zip(np.atleast_1d(axes), sets):
            if "oracle" in out and s in out["oracle"]:
                ax.axhline(out["oracle"][s], color="#6b6157", lw=1, ls=":")
        fig.suptitle(f"Point-Goal: return vs. post-training steps (mean ± std over {len(out['seeds'])} seeds; dotted = oracle)", fontsize=11)
        fig.tight_layout(); fig.savefig(FIGS / f"adaptation{sfx}.png", dpi=150); plt.close(fig)
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
        fig.tight_layout(); fig.savefig(FIGS / f"waypoints{sfx}.png", dpi=150); plt.close(fig)
    if out.get("axis_c"):
        c = out["axis_c"]
        names = [v for v in V["cont"] if v in c]
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
        for ax, key, title in ((axes[0], "final", "final mean return (higher is better)"),
                               (axes[1], "forgetting", "forgetting (lower is better)")):
            ax.barh([LABELS[n] for n in names], [c[n][key][0] for n in names], xerr=[c[n][key][1] for n in names],
                    color=[COLORS[n] for n in names], alpha=0.85)
            ax.set_title(title, fontsize=10); ax.invert_yaxis()
        axes[1].set_yticklabels([])
        fig.suptitle("Continual post-training over 5 out-of-distribution tasks", fontsize=11)
        fig.tight_layout(); fig.savefig(FIGS / f"continual{sfx}.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--version", default="v1", choices=list(VERSIONS))
    ap.add_argument("--seeds", type=int, nargs="+")
    a = ap.parse_args()
    V = VERSIONS[a.version]
    out = build(Path(a.results), a.seeds or V["seeds"], a.version)
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / V["out"]).write_text(json.dumps(out))
    figures(out, V)
    for v in out["verdicts"]:
        print(f"axis {v['axis']} {v['set']:>17} {LABELS[V['hero']]} vs {LABELS[v['vs']]:<26} {v['mehta'][0]:8.3f}±{v['mehta'][1]:.3f} vs "
              f"{v['other'][0]:8.3f}±{v['other'][1]:.3f} -> {v['verdict']}")


if __name__ == "__main__":
    main()
