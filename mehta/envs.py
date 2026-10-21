"""Small, fast, pure-NumPy meta-RL environments (level 1 of the fidelity ladder).

Point-Goal is modeled on PEARL's sparse point robot
(github.com/katerakelly/oyster, rlkit/envs/point_robot.py): start at the origin,
actions in [-0.1, 0.1]^2, goals on the unit circle, horizon 20. Differences that
are ours (not PEARL's): a dense reward -||p - g|| everywhere, goals on the FULL
circle so we can hold the lower half out as out-of-distribution, and an optional
action rotation that stands in for a sim-to-real dynamics mismatch.

Point-Waypoints is ours: a long, sparse, three-stage version of the same task.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STEP = 0.1          # max displacement per axis per step (PEARL: action box +-0.1)
GOAL_RADIUS = 0.2   # PEARL's sparse goal radius


@dataclass(frozen=True)
class Task:
    angle: float                 # goal angle on the unit circle (radians)
    rotation: float = 0.0        # action rotation in radians (dynamics shift)
    name: str = ""

    def as_dict(self):
        return {"angle": self.angle, "rotation": self.rotation, "name": self.name}


def _rot(a: np.ndarray, theta: float) -> np.ndarray:
    if theta == 0.0:
        return a
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * a[0] - s * a[1], s * a[0] + c * a[1]])


class PointGoal:
    """Reach a goal on the unit circle. Dense reward = -distance."""

    obs_dim = 2
    act_dim = 2
    horizon = 20
    gamma = 0.9            # PEARL's discount for the point robot

    def __init__(self, task: Task):
        self.task = task
        self.goal = np.array([np.cos(task.angle), np.sin(task.angle)])
        self.pos = np.zeros(2)
        self.t = 0

    def reset(self) -> np.ndarray:
        self.pos = np.zeros(2)
        self.t = 0
        return self.pos.copy()

    def step(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0) * STEP
        self.pos = self.pos + _rot(a, self.task.rotation)
        self.t += 1
        r = -float(np.linalg.norm(self.pos - self.goal))
        # (train reward, context/eval reward): identical for the dense env
        return self.pos.copy(), r, r, self.t >= self.horizon


class PointWaypoints:
    """Visit three waypoints in order; the task is the first waypoint's angle.

    Waypoint k sits on the unit circle at angle + k * pi/4. The evaluation /
    context reward is SPARSE: +1 the step a waypoint is reached (then the next one
    becomes active), 0 otherwise, max 3 per episode. Following PEARL's sparse
    point-robot protocol, the critic is meta-trained with a dense shaping reward
    (-distance to the active waypoint, plus the sparse bonus) while the encoder
    only ever sees the sparse reward; post-training on test tasks uses ONLY the
    sparse reward.
    """

    n_way = 3
    obs_dim = 2 + n_way + 1    # position + one-hot of the active stage (3 = done)
    act_dim = 2
    horizon = 60
    gamma = 0.95
    spacing = np.pi / 4

    def __init__(self, task: Task):
        self.task = task
        ang = task.angle + self.spacing * np.arange(self.n_way)
        self.waypoints = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        self.reset()

    def _obs(self):
        onehot = np.zeros(self.n_way + 1)
        onehot[self.stage] = 1.0
        return np.concatenate([self.pos, onehot])

    def reset(self) -> np.ndarray:
        self.pos = np.zeros(2)
        self.stage = 0
        self.t = 0
        return self._obs()

    def step(self, action: np.ndarray):
        a = np.clip(action, -1.0, 1.0) * STEP
        self.pos = self.pos + _rot(a, self.task.rotation)
        self.t += 1
        sparse, dense = 0.0, 0.0
        if self.stage < self.n_way:
            d = float(np.linalg.norm(self.pos - self.waypoints[self.stage]))
            dense = -d
            if d <= GOAL_RADIUS:
                sparse = 1.0
                self.stage += 1
        return self._obs(), dense + sparse, sparse, self.t >= self.horizon


ENVS = {"point_goal": PointGoal, "waypoints": PointWaypoints}


@dataclass
class TaskSets:
    train: list[Task]
    evals: dict[str, list[Task]] = field(default_factory=dict)


def make_task_sets(env_name: str, n_train: int = 40, n_eval: int = 5, seed: int = 0) -> TaskSets:
    """Fixed task splits. Train goals: angle in [0, pi]. Eval tasks are drawn with
    a separate RNG so they never coincide with training goals."""
    train = [Task(float(a), 0.0, f"train{i}") for i, a in enumerate(np.linspace(0, np.pi, n_train))]
    rng = np.random.default_rng(1000 + seed)
    id_angles = rng.uniform(0, np.pi, n_eval)
    evals = {"id": [Task(float(a), 0.0, f"id{i}") for i, a in enumerate(id_angles)]}
    if env_name == "point_goal":
        ood = rng.uniform(np.pi + 0.15, 2 * np.pi - 0.15, n_eval)
        evals["ood_goal"] = [Task(float(a), 0.0, f"oodg{i}") for i, a in enumerate(ood)]
        evals["ood_dyn"] = [Task(float(a), np.deg2rad(60.0), f"oodd{i}") for i, a in enumerate(id_angles)]
        # a separate held-out continual sequence (lower half circle)
        seq = rng.uniform(np.pi + 0.15, 2 * np.pi - 0.15, n_eval)
        evals["continual"] = [Task(float(a), 0.0, f"seq{i}") for i, a in enumerate(seq)]
    # ID validation tasks for (dev-seed-only) tuning, disjoint from the test draws
    val = np.random.default_rng(2000 + seed).uniform(0, np.pi, n_eval)
    evals["val"] = [Task(float(a), 0.0, f"val{i}") for i, a in enumerate(val)]
    return TaskSets(train, evals)
