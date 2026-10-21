"""Export the seed-0 Point-Goal models to site/data/demo.json for the in-browser demo:
meta-init actor + encoder weights, post-trained actor snapshots, the latent-space
explorer, and a parity fixture that tests/test_export.py checks against the JS.

  uv run python -m mehta.export_demo
"""
from __future__ import annotations

import json

import numpy as np
import torch

from .agent import Agent, SACConfig
from .buffers import to_context
from .envs import PointGoal, Task, make_task_sets
from .experiments import RESULTS
from .report import SITE_DATA
from .rollout import rollout, sample_z

MODELS = RESULTS / "models"
DEMO_METHODS = {"mehta": "mehta", "pearl": "pearl"}          # model file -> label
ADAPTED = {"mehta": "mehta", "pearl_ft": "pearl"}             # variant -> source model
N_ADAPTED_TASKS = 3


def layers(state: dict, prefix: str = "net."):
    out, i = [], 0
    while f"{prefix}{i}.weight" in state:
        out.append({"W": np.round(state[f"{prefix}{i}.weight"].numpy(), 5).tolist(),
                    "b": np.round(state[f"{prefix}{i}.bias"].numpy(), 5).tolist()})
        i += 2
    return out


def load_agent(name: str) -> Agent:
    ck = torch.load(MODELS / f"point_goal_{name}_s0.pt")
    ag = Agent(PointGoal.obs_dim, PointGoal.act_dim, SACConfig(hidden=ck["hidden"], latent_dim=ck["latent_dim"]))
    ag.actor.load_state_dict(ck["actor"]); ag.encoder.load_state_dict(ck["encoder"])
    return ag


@torch.no_grad()
def latent_paths(ag: Agent, n_angles: int = 36, seed: int = 0):
    """Posterior mean after every transition of two exploration episodes, for
    goals all around the circle; projected to 2-D with PCA over all points."""
    torch.manual_seed(seed)
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    raw = []
    for ang in angles:
        env, ctx, path = PointGoal(Task(float(ang))), torch.zeros(0, ag.ctx_dim), []
        for _ in range(2):
            p = rollout(env, ag, sample_z(ag, ctx))
            for t in range(len(p["r"])):
                ctx = torch.cat([ctx, to_context(p["o"][t:t + 1], p["a"][t:t + 1], p["rc"][t:t + 1], p["o2"][t:t + 1])])
                mu, var = ag.posterior(ctx)
                path.append((mu.numpy(), float(var.mean())))
        raw.append(path)
    X = np.concatenate([[m for m, _ in p] for p in raw])
    mean = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    explained = (S[:2] ** 2).sum() / (S ** 2).sum()
    paths = [[[*np.round((m - mean) @ Vt[:2].T, 4).tolist(), round(v, 5)] for m, v in p] for p in raw]
    return {"angles": np.round(angles, 4).tolist(), "paths": paths, "explained": round(float(explained), 3)}


def main():
    demo = {"env": {"step": 0.1, "horizon": PointGoal.horizon, "goal_radius": 0.2}, "models": {}, "adapted": {}, "latent": {}}
    for name, label in DEMO_METHODS.items():
        ag = load_agent(name)
        demo["models"][label] = {"actor": layers(ag.actor.state_dict()), "encoder": layers(ag.encoder.state_dict()),
                                 "latent_dim": ag.latent_dim}
        demo["latent"][label] = latent_paths(ag)
    ood = make_task_sets("point_goal", seed=0).evals["ood_goal"]
    for variant in ADAPTED:
        demo["adapted"][variant] = []
        for ti in range(N_ADAPTED_TASKS):
            kept = torch.load(MODELS / f"point_goal_{variant}_s0_ood_goal{ti}_adapted.pt")
            demo["adapted"][variant].append({
                "task": ood[ti].as_dict(),
                "K": {str(k): {"actor": layers(v["actor"]), "z": np.round(v["z"].numpy(), 5).tolist()} for k, v in kept.items()}})
    # parity fixture: Python reference outputs the JS must reproduce
    ag = load_agent("mehta")
    g = torch.Generator().manual_seed(0)
    obs, z = torch.randn(4, 2, generator=g), torch.randn(4, ag.latent_dim, generator=g)
    ctx = torch.randn(6, ag.ctx_dim, generator=g)
    with torch.no_grad():
        act, _ = ag.actor(obs, z, deterministic=True)
        mu, var = ag.posterior(ctx)
    demo["parity"] = {"obs": obs.tolist(), "z": z.tolist(), "action": act.tolist(), "ctx": ctx.tolist(),
                      "mu": mu.tolist(), "var": var.tolist()}
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / "demo.json").write_text(json.dumps(demo, separators=(",", ":")))
    print("wrote", SITE_DATA / "demo.json", round((SITE_DATA / "demo.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
