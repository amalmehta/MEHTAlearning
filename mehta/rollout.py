"""Episode collection."""
from __future__ import annotations

import numpy as np
import torch

from .agent import Agent


def rollout(env, agent: Agent, z: torch.Tensor, deterministic: bool = False, record: bool = False) -> dict:
    o = env.reset()
    keys = ("o", "a", "r", "rc", "o2")
    path = {k: [] for k in keys}
    done = False
    while not done:
        a = agent.act(o, z, deterministic)
        o2, r, rc, done = env.step(a)
        for k, v in zip(keys, (o, a, r, rc, o2)):
            path[k].append(v)
        o = o2
    path = {k: np.asarray(v, np.float32) for k, v in path.items()}
    path["ret"] = float(path["rc"].sum())          # evaluation return uses the context/eval reward
    if record:
        path["xy"] = np.concatenate([[[0.0, 0.0]], path["o2"][:, :2]]).round(4).tolist()
    return path


def sample_z(agent: Agent, ctx: torch.Tensor, mean: bool = False) -> torch.Tensor:
    mu, var = agent.posterior(ctx)
    return mu if mean else mu + var.sqrt() * torch.randn_like(mu)
