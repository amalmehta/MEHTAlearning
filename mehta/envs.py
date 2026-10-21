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
    gain: float = 1.0            # action gain (training-time perturbation only)
    drift: tuple = (0.0, 0.0)    # constant displacement per step (training-time perturbation only)
    mirror: bool = False         # flip the x component of the action (held-out shift only)

    def as_dict(self):
        return {"angle": self.angle, "rotation": self.rotation, "name": self.name,
                "gain": self.gain, "drift": list(self.drift), "mirror": self.mirror}


def augment(task: Task, rng: np.random.Generator) -> Task:
    """Random dynamics perturbation used during meta-training (MEHTA v2's inner
    loop, PEARL-DR's data). Deliberately a different family from the test shifts
    (held-out goals, 60-degree action rotation): gain in [0.5, 1.5] and a constant
    drift of up to 0.03 per step (30% of the max step) in a random direction."""
    phi, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.03)
    return Task(task.angle, task.rotation, task.name + "+aug", float(rng.uniform(0.5, 1.5)),
                (float(r * np.cos(phi)), float(r * np.sin(phi))))


def _move(pos, action, task: Task):
    a = np.clip(action, -1.0, 1.0) * STEP * task.gain
    if task.mirror:
        a = np.array([-a[0], a[1]])
    return pos + _rot(a, task.rotation) + np.asarray(task.drift)


def oracle_return(task: Task) -> float:
    """Reference return (an estimate, not a proven optimum) of a greedy controller
    that knows the goal AND the dynamics: each step it computes the action that
    would land on the goal (net of drift, inverting the action transform), then
    tries two ways of fitting it into the action box (per-axis clipping and
    direction-preserving scaling) and keeps whichever ends closer to the goal."""
    env = PointGoal(task)
    o, ret, done = env.reset(), 0.0, False
    while not done:
        want = env.goal - o - np.asarray(task.drift)
        a = _rot(want, -task.rotation) / (STEP * task.gain)
        if task.mirror:
            a = np.array([-a[0], a[1]])
        cands = [np.clip(a, -1, 1), a / max(1.0, float(np.abs(a).max()))]
        a = min(cands, key=lambda c: np.linalg.norm(_move(o, c, task) - env.goal))
        o, r, _, done = env.step(a)
        ret += r
    return ret


# Held-out shifts for v3 (pre-registered in MEHTAlearning.md). None is ever used in
# training: training perturbations are gain 0.5-1.5 and drift <= 0.03 only.
SHIFTS = {
    "shift_rot120": dict(rotation=np.deg2rad(120.0)),
    "shift_invert": dict(rotation=np.pi),
    "shift_mirror": dict(mirror=True),
    "shift_wind": dict(wind=0.06),
    "shift_ood_rot60": dict(rotation=np.deg2rad(60.0), ood=True),
}


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
        self.pos = _move(self.pos, action, self.task)
        self.t += 1
        r = -float(np.linalg.norm(self.pos - self.goal))
        # (train reward, context/eval reward): identical for the dense env
        return self.pos.copy(), r, r, self.t >= self.horizon


class PointWaypoints:
    """Visit three waypoints in order; the task is the first waypoint's angle.

    Waypoint k sits on the unit circle at angle + k * pi/4. The evaluation /
    context reward is SPARSE: +1 the step a waypoint is reached (then the next one
    becomes active), 0 otherwise, max 3 per episode. Following PEARL's sparse
    point-robot protocol, the critic is meta-trained with a shaped reward while
    the encoder only ever sees the sparse reward; post-training on test tasks
    uses ONLY the sparse reward.

    Shaping (v2) is potential-based (Ng, Harada & Russell 1999), so it cannot
    change the optimal policy: r_train = sparse + gamma*Phi(s') - Phi(s), with
    Phi = -(distance to the active waypoint + chord length * waypoints left after
    it), and Phi = 0 once all three are reached. v1 used -distance + sparse,
    whose jump at each waypoint made hovering short of it about as good as
    finishing; that trap is gone here.
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

    def potential(self) -> float:
        if self.stage >= self.n_way:
            return 0.0
        chord = 2 * np.sin(self.spacing / 2)                      # distance between consecutive waypoints
        d = float(np.linalg.norm(self.pos - self.waypoints[self.stage]))
        return -(d + chord * (self.n_way - 1 - self.stage))

    def step(self, action: np.ndarray):
        phi0 = self.potential()
        self.pos = _move(self.pos, action, self.task)
        self.t += 1
        sparse = 0.0
        if self.stage < self.n_way and np.linalg.norm(self.pos - self.waypoints[self.stage]) <= GOAL_RADIUS:
            sparse = 1.0
            self.stage += 1
        shaped = sparse + self.gamma * self.potential() - phi0
        return self._obs(), shaped, sparse, self.t >= self.horizon


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
    if env_name == "point_goal":
        # v3 shifts: their own RNG, so earlier task draws are unchanged
        srng = np.random.default_rng(3000 + seed)
        s_id = srng.uniform(0, np.pi, n_eval)
        s_ood = srng.uniform(np.pi + 0.15, 2 * np.pi - 0.15, n_eval)
        wind_dir = srng.uniform(0, 2 * np.pi, n_eval)
        for name, spec in SHIFTS.items():
            angles = s_ood if spec.get("ood") else s_id
            evals[name] = [
                Task(float(a), float(spec.get("rotation", 0.0)), f"{name}{i}",
                     drift=(float(spec["wind"] * np.cos(wind_dir[i])), float(spec["wind"] * np.sin(wind_dir[i])))
                     if "wind" in spec else (0.0, 0.0),
                     mirror=bool(spec.get("mirror", False)))
                for i, a in enumerate(angles)]
    # ID validation tasks for (dev-seed-only) tuning, disjoint from the test draws
    val = np.random.default_rng(2000 + seed).uniform(0, np.pi, n_eval)
    evals["val"] = [Task(float(a), 0.0, f"val{i}") for i, a in enumerate(val)]
    return TaskSets(train, evals)
