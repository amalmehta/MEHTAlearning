"""Meta-World ML1 (level 1.5 of the fidelity ladder): Sawyer-arm manipulation,
between the 2-D point tasks and full level-2 robotics runs.

Each family is one ML1 benchmark (reach-v3, push-v3, pick-place-v3). A task is
one of the benchmark's goal (and, for push/pick-place, object) positions. The
goal is NOT in the observation (ML1 zeroes it), so it has to be inferred from
reward, as in PEARL's Meta-World runs. Splits, fixed across seeds:
  train  the benchmark's train goals 0..39
  val    its train goals 45..49 (dev-only tuning, never trained on)
  id     5 of its 50 held-out test goals, drawn per seed

The wrapper exposes the same interface as envs.PointGoal: class attributes
obs_dim / act_dim / horizon / gamma, Env(task), reset() -> obs and
step(a) -> (obs, train reward, context/eval reward, done). Both rewards are
Meta-World's dense v3 reward scaled to [0, 1] per step; success (the standard
Meta-World metric) is tracked on the env as `success` but not yet logged.

Episodes are 200 steps with gamma 0.99 (effective horizon ~100), 10x the point
tasks: push and pick-place are reach -> grasp/contact -> move, so reward for
the final stage depends on actions taken ~50-150 steps earlier.

Needs the optional dependency group: uv sync --group metaworld
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

BENCH_SEED = 0          # ML1's own seed: fixes the goal positions, identical for every run seed
N_TRAIN, VAL = 40, range(45, 50)


@dataclass(frozen=True)
class MWTask:
    benchmark: str      # e.g. "push-v3"
    split: str          # "train" (ML1 train goals) or "test" (ML1 held-out goals)
    index: int
    name: str = ""

    def as_dict(self):
        return {"benchmark": self.benchmark, "split": self.split, "index": self.index, "name": self.name,
                "goal": [round(float(x), 4) for x in _rand_vec(self)[-3:]]}


@lru_cache(maxsize=None)
def _ml1(benchmark: str):
    import metaworld
    return metaworld.ML1(benchmark, seed=BENCH_SEED)


def _mw_task(t: MWTask):
    ml1 = _ml1(t.benchmark)
    return (ml1.train_tasks if t.split == "train" else ml1.test_tasks)[t.index]


def _rand_vec(t: MWTask) -> np.ndarray:
    import pickle
    return np.asarray(pickle.loads(_mw_task(t).data)["rand_vec"])


class _Sim:
    """One MuJoCo env per benchmark per process (construction takes ~1 s); the
    task it currently holds is swapped in with set_task() on reset."""

    _cache: dict[str, "_Sim"] = {}

    def __init__(self, benchmark: str):
        self.env = _ml1(benchmark).train_classes[benchmark]()
        self.task: MWTask | None = None

    @classmethod
    def get(cls, benchmark: str) -> "_Sim":
        if benchmark not in cls._cache:
            cls._cache[benchmark] = cls(benchmark)
        return cls._cache[benchmark]


class MetaWorldML1:
    benchmark = ""          # set per family below
    obs_dim = 39            # hand (4), object (14), previous frame (18), goal (3, zeroed by ML1)
    act_dim = 4
    horizon = 200
    gamma = 0.99
    reward_scale = 0.1      # v3 rewards are in [0, 10] per step

    def __init__(self, task: MWTask):
        assert task.benchmark == self.benchmark, (task.benchmark, self.benchmark)
        self.task = task
        self.sim = _Sim.get(task.benchmark)
        self.t, self.success = 0, False

    def reset(self) -> np.ndarray:
        if self.sim.task != self.task:
            self.sim.env.set_task(_mw_task(self.task))
            self.sim.task = self.task
        o, _ = self.sim.env.reset()
        self.t, self.success = 0, False
        return o

    def step(self, action: np.ndarray):
        o, r, _, _, info = self.sim.env.step(np.clip(action, -1.0, 1.0).astype(np.float32))
        self.t += 1
        self.success |= bool(info["success"])
        r = float(r) * self.reward_scale
        return o, r, r, self.t >= self.horizon


def _family(benchmark: str):
    return type(f"MW_{benchmark}", (MetaWorldML1,), {"benchmark": benchmark})


MW_ENVS = {
    "mw_reach": _family("reach-v3"),
    "mw_push": _family("push-v3"),
    "mw_pick_place": _family("pick-place-v3"),
}


def mw_task_sets(env_name: str, n_eval: int = 5, seed: int = 0):
    """(train tasks, eval sets) for a Meta-World family; see the module docstring."""
    b = MW_ENVS[env_name].benchmark
    train = [MWTask(b, "train", i, f"train{i}") for i in range(N_TRAIN)]
    test_idx = np.random.default_rng(1000 + seed).choice(50, n_eval, replace=False)
    evals = {"id": [MWTask(b, "test", int(i), f"id{j}") for j, i in enumerate(test_idx)],
             "val": [MWTask(b, "train", i, f"val{j}") for j, i in enumerate(VAL)]}
    return train, evals
