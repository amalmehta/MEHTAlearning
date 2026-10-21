"""Per-task replay buffers. Each transition stores two rewards: `r` (what the critic
trains on during meta-training) and `rc` (what the encoder and evaluation see)."""
from __future__ import annotations

import numpy as np
import torch


class TaskBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 100_000):
        self.o = np.zeros((capacity, obs_dim), np.float32)
        self.a = np.zeros((capacity, act_dim), np.float32)
        self.r = np.zeros(capacity, np.float32)
        self.rc = np.zeros(capacity, np.float32)
        self.o2 = np.zeros((capacity, obs_dim), np.float32)
        self.cap, self.n, self.i = capacity, 0, 0

    def __len__(self):
        return self.n

    def clear(self):
        self.n = self.i = 0

    def add_path(self, path: dict):
        for k in range(len(path["r"])):
            j = self.i
            self.o[j], self.a[j], self.r[j] = path["o"][k], path["a"][k], path["r"][k]
            self.rc[j], self.o2[j] = path["rc"][k], path["o2"][k]
            self.i = (self.i + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, rng: np.random.Generator, batch: int, reward_key: str = "r"):
        idx = rng.integers(0, self.n, batch)
        t = lambda x: torch.from_numpy(x[idx])
        return t(self.o), t(self.a), t(getattr(self, reward_key)), t(self.o2)

    def context(self, rng: np.random.Generator | None = None, size: int | None = None) -> torch.Tensor:
        """(o, a, rc, o') rows. With `size`, a random subset; else everything."""
        idx = np.arange(self.n) if size is None else rng.integers(0, self.n, size)
        return to_context(self.o[idx], self.a[idx], self.rc[idx], self.o2[idx])


def to_context(o, a, rc, o2) -> torch.Tensor:
    return torch.from_numpy(
        np.concatenate([o, a, np.asarray(rc, np.float32)[:, None], o2], axis=1).astype(np.float32)
    )
