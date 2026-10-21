"""Domain randomization and held-out physics shifts, as an environment wrapper.

Five physical parameters, each with a nominal value and a TRAINING range:

  parameter   nominal   training range       scale
  mass        1.0       [0.8, 1.25]          log
  friction    1.0       [0.8, 1.25]          log
  gain        1.0       [0.8, 1.25]          log (actuator gain)
  obs_noise   0.0       [0, 0.02]            std of Gaussian noise on continuous observations
  delay       0         {0, 1}               steps of actuation latency

Meta-training draws a fresh parameter set every time an environment is built
(log-uniform / uniform / integer-uniform inside the training range), so the
dynamics stay fixed within one task's context but vary across visits.

Evaluation uses a SHIFT MAGNITUDE s >= 0 instead: s = 0 is nominal, s = 1 puts
every parameter at the edge of its training range, s > 1 is outside it. For log
parameters the log-deviation is s times the training half-width (s = 2 gives
mass 0.64 or 1.56); noise is s * 0.02 and delay is round(s * 1). Each held-out
task gets a fixed random sign per log parameter, so a task's direction of shift
is the same at every magnitude and only the size changes.

How the parameters act depends on the wrapped environment:
  * Point envs (anything whose `task` is an envs.Task): the wrapper is the
    actuator. A point mass with viscous friction, integrated per step:
        v <- (1 - d) v + d (gain / friction) a,   d = min(1, friction / mass)
    so steady speed is gain * a / friction and the response time grows with
    mass / friction. At nominal values d = 1 and v = a exactly, which is the
    unwrapped environment. Speed is capped at VMAX times the nominal max step.
  * MuJoCo envs (anything exposing `env.model` with body_mass, body_inertia,
    geom_friction and actuator_gear, e.g. a mujoco.MjModel): the model arrays
    are scaled from their nominal values. No MuJoCo task exists in this repo
    yet; the adapter is exercised in tests with a stand-in model.
Delay and observation noise are applied the same way for every environment.
The reward is always computed by the wrapped env from the true state.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .envs import Task

VMAX = 3.0

# name: (nominal, low, high, kind)
PARAMS = {
    "mass": (1.0, 0.8, 1.25, "log"),
    "friction": (1.0, 0.8, 1.25, "log"),
    "gain": (1.0, 0.8, 1.25, "log"),
    "obs_noise": (0.0, 0.0, 0.02, "lin"),
    "delay": (0, 0, 1, "int"),
}
LOG_PARAMS = [k for k, v in PARAMS.items() if v[3] == "log"]
MAGNITUDES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]        # s <= 1 inside the training range, s > 1 outside


@dataclass(frozen=True)
class Physics:
    mass: float = 1.0
    friction: float = 1.0
    gain: float = 1.0
    obs_noise: float = 0.0
    delay: int = 0

    def as_dict(self):
        return asdict(self)

    def in_training_range(self) -> bool:
        return all(PARAMS[k][1] - 1e-9 <= v <= PARAMS[k][2] + 1e-9 for k, v in asdict(self).items())


NOMINAL = Physics()


def sample_training(rng: np.random.Generator) -> Physics:
    """One draw from the training range."""
    vals = {}
    for k, (_, lo, hi, kind) in PARAMS.items():
        if kind == "log":
            vals[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "lin":
            vals[k] = float(rng.uniform(lo, hi))
        else:
            vals[k] = int(rng.integers(lo, hi + 1))
    return Physics(**vals)


def shifted(s: float, signs: dict[str, int]) -> Physics:
    """Held-out physics at shift magnitude s; `signs` gives +1 / -1 per log parameter."""
    vals = {}
    for k, (nom, lo, hi, kind) in PARAMS.items():
        if kind == "log":
            edge = hi if signs[k] > 0 else lo
            vals[k] = float(nom * (edge / nom) ** s)
        elif kind == "lin":
            vals[k] = float(nom + s * (hi - nom))
        else:
            vals[k] = int(round(nom + s * (hi - nom)))
    return Physics(**vals)


def shift_signs(n: int, seed: int) -> list[dict[str, int]]:
    """Fixed per-task shift directions, from their own RNG."""
    rng = np.random.default_rng(5000 + seed)
    return [{k: int(rng.choice([-1, 1])) for k in LOG_PARAMS} for _ in range(n)]


# ---- physics adapters ---------------------------------------------------------
class _PointMass:
    def __init__(self, env, p: Physics):
        self.p = p
        # the base env clips actions to [-1, 1]; give it VMAX headroom so the
        # wrapper's velocity, not the base clip, is what moves the point
        env.task = replace(env.task, gain=env.task.gain * VMAX)
        self.v = np.zeros(env.act_dim)

    def reset(self):
        self.v[:] = 0.0

    def __call__(self, a):
        d = min(1.0, self.p.friction / self.p.mass)
        self.v = np.clip((1 - d) * self.v + d * (self.p.gain / self.p.friction) * a, -VMAX, VMAX)
        return self.v / VMAX


class _Mujoco:
    FIELDS = ("body_mass", "body_inertia", "geom_friction", "actuator_gear")

    def __init__(self, env, p: Physics):
        m = env.model
        if not hasattr(env, "_dr_nominal"):          # remember the unscaled model once per env
            env._dr_nominal = {f: np.array(getattr(m, f), copy=True) for f in self.FIELDS}
        nom = env._dr_nominal
        m.body_mass[:] = nom["body_mass"] * p.mass
        m.body_inertia[:] = nom["body_inertia"] * p.mass
        m.geom_friction[:, 0] = nom["geom_friction"][:, 0] * p.friction     # sliding friction
        m.actuator_gear[:, 0] = nom["actuator_gear"][:, 0] * p.gain

    def reset(self):
        pass

    def __call__(self, a):
        return a


def _physics_for(env, p: Physics):
    if hasattr(env, "model") and all(hasattr(env.model, f) for f in _Mujoco.FIELDS):
        return _Mujoco(env, p)
    if isinstance(getattr(env, "task", None), Task):
        return _PointMass(env, p)
    raise TypeError(f"{type(env).__name__}: need a MuJoCo `model` or an envs.Task `task` to apply physics")


# ---- the wrapper ----------------------------------------------------------------
class PhysicsWrapper:
    """Wraps one environment instance with fixed physics `p`. `seed` drives the
    observation noise, so the same (p, seed) replays the same episodes."""

    def __init__(self, env, p: Physics, seed: int = 0):
        self.env, self.p = env, p
        self.rng = np.random.default_rng(seed)
        self.phys = _physics_for(env, p)
        self.noisy = getattr(env, "noisy_obs_dims", env.obs_dim)
        self.queue = []

    def __getattr__(self, name):                      # obs_dim, horizon, gamma, task, ...
        return getattr(self.env, name)

    def _obs(self, o):
        if self.p.obs_noise > 0:
            o = o.copy()
            o[: self.noisy] += self.rng.normal(0.0, self.p.obs_noise, self.noisy)
        return o

    def reset(self):
        self.phys.reset()
        self.queue = [np.zeros(self.env.act_dim)] * self.p.delay
        return self._obs(self.env.reset())

    def step(self, action):
        self.queue.append(np.clip(np.asarray(action, float), -1.0, 1.0))
        o, r, rc, done = self.env.step(self.phys(self.queue.pop(0)))
        return self._obs(o), r, rc, done


class Randomized:
    """Stands in for an env class (`Randomized(PointGoal, seed=0)(task)`), so it
    plugs into meta_train, TaskState and post_train unchanged.

    physics=None: every built env draws fresh physics from the training range.
    physics=Physics(...): every built env uses exactly that physics.
    The same seed always yields the same sequence of physics and noise."""

    def __init__(self, Env, seed: int, physics: Physics | None = None):
        self.Env, self.physics = Env, physics
        self.rng = np.random.default_rng(seed)
        self.obs_dim, self.act_dim, self.gamma, self.horizon = Env.obs_dim, Env.act_dim, Env.gamma, Env.horizon

    def __call__(self, task) -> PhysicsWrapper:
        p = self.physics or sample_training(self.rng)
        return PhysicsWrapper(self.Env(task), p, seed=int(self.rng.integers(2**31)))
