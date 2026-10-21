from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mehta.agent import Agent, SACConfig
from mehta.envs import PointGoal, PointWaypoints, Task
from mehta.randomize import (MAGNITUDES, NOMINAL, PARAMS, Physics, PhysicsWrapper, Randomized, shift_signs,
                             shifted)


def _episode(env, actions):
    obs = [env.reset()]
    for a in actions:
        obs.append(env.step(a)[0])
    return np.array(obs)


@pytest.mark.parametrize("Env", [PointGoal, PointWaypoints])
def test_same_seed_same_physics_and_noise(Env):
    """The deterministic guarantee: same seed -> same physics draws and the same
    noisy observations; a different seed -> different draws."""
    actions = np.random.default_rng(0).uniform(-1, 1, (Env.horizon, 2))
    runs = {}
    for name, seed in (("a", 3), ("b", 3), ("c", 4)):
        R = Randomized(Env, seed=seed)
        envs = [R(Task(0.5)) for _ in range(5)]
        runs[name] = ([e.p for e in envs], [_episode(e, actions) for e in envs])
    assert runs["a"][0] == runs["b"][0]
    assert all(np.array_equal(x, y) for x, y in zip(runs["a"][1], runs["b"][1]))
    assert runs["a"][0] != runs["c"][0]
    assert all(p.in_training_range() for p in runs["a"][0])


@pytest.mark.parametrize("Env", [PointGoal, PointWaypoints])
def test_nominal_physics_is_the_unwrapped_env(Env):
    actions = np.random.default_rng(1).uniform(-1.5, 1.5, (Env.horizon, 2))
    base, wrapped = Env(Task(0.7)), PhysicsWrapper(Env(Task(0.7)), NOMINAL)
    assert np.allclose(_episode(base, actions), _episode(wrapped, actions))


def test_delay_shifts_actions_by_k_steps():
    env = PhysicsWrapper(PointGoal(Task(0.0)), Physics(delay=2))
    env.reset()
    xs = [env.step(np.array([1.0, 0.0]))[0][0] for _ in range(4)]
    assert np.allclose(xs, [0.0, 0.0, 0.1, 0.2])


def test_heavier_mass_responds_slower_and_lower_friction_coasts():
    heavy = PhysicsWrapper(PointGoal(Task(0.0)), Physics(mass=2.0))
    heavy.reset()
    assert heavy.step(np.array([1.0, 0.0]))[0][0] == pytest.approx(0.05)        # d = 0.5
    slick = PhysicsWrapper(PointGoal(Task(0.0)), Physics(friction=0.5))
    slick.reset()
    slick.step(np.array([1.0, 0.0]))
    x1 = slick.pos[0]
    slick.step(np.zeros(2))                                                      # keeps moving with zero action
    assert slick.pos[0] > x1


def test_noise_touches_only_continuous_dims_and_not_the_reward():
    env = PhysicsWrapper(PointWaypoints(Task(0.0)), Physics(obs_noise=0.05), seed=0)
    o = env.reset()
    assert not np.allclose(o[:2], 0) and np.array_equal(o[2:], [1, 0, 0, 0])
    _, r, rc, _ = env.step(np.zeros(2))
    clean = PointWaypoints(Task(0.0)); clean.reset()
    assert (r, rc) == clean.step(np.zeros(2))[1:3]


def test_shift_magnitudes_inside_and_outside_training_range():
    signs = shift_signs(5, seed=0)
    assert shift_signs(5, seed=0) == signs
    for sg in signs:
        assert shifted(0.0, sg) == NOMINAL
        for s in MAGNITUDES:
            assert shifted(s, sg).in_training_range() == (s <= 1.0)
        edge = shifted(1.0, sg)
        assert {edge.mass, edge.friction, edge.gain} <= {PARAMS["mass"][1], PARAMS["mass"][2]}


def test_mujoco_adapter_scales_model_from_nominal_every_time():
    model = SimpleNamespace(body_mass=np.array([1.0, 2.0]), body_inertia=np.ones((2, 3)),
                            geom_friction=np.array([[1.0, 0.005, 1e-4]]), actuator_gear=np.array([[100.0, 0, 0, 0, 0, 0]]))
    env = SimpleNamespace(model=model, obs_dim=3, act_dim=1, reset=lambda: np.zeros(3),
                          step=lambda a: (np.zeros(3), 0.0, 0.0, False))
    PhysicsWrapper(env, Physics(mass=2.0, friction=0.5, gain=1.5))
    PhysicsWrapper(env, Physics(mass=2.0, friction=0.5, gain=1.5))       # re-wrapping must not compound
    assert np.allclose(model.body_mass, [2.0, 4.0]) and np.allclose(model.body_inertia, 2.0)
    assert np.allclose(model.geom_friction, [[0.5, 0.005, 1e-4]]) and model.actuator_gear[0, 0] == 150.0


def test_wrapper_rejects_unknown_envs():
    with pytest.raises(TypeError):
        PhysicsWrapper(SimpleNamespace(obs_dim=1, act_dim=1), NOMINAL)


def test_checkpoint_round_trip(tmp_path):
    ag = Agent(PointGoal.obs_dim, PointGoal.act_dim, SACConfig(hidden=8))
    torch.save(ag.checkpoint(), tmp_path / "a.pt")
    back = Agent.from_checkpoint(torch.load(tmp_path / "a.pt", weights_only=False))
    for n in Agent.NETS:
        for p, q in zip(getattr(ag, n).parameters(), getattr(back, n).parameters()):
            assert torch.equal(p, q)
    assert float(back.log_alpha) == pytest.approx(float(ag.log_alpha))


def test_phys_methods_train_through_the_wrapper():
    from mehta.train import TrainConfig, meta_train
    _, info = meta_train("waypoints", "mehta_phys", 0, TrainConfig(n_iters=1, eval_every=10**6, grad_steps=2), SACConfig(hidden=8))
    assert info["grad_steps"] == 2 + 50 * 2
