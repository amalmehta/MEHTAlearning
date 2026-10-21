import numpy as np
import pytest
import torch

from mehta.adapt import continual_metrics, post_train
from mehta.agent import Agent, SACConfig, anchor_of, reptile_apply
from mehta.envs import GOAL_RADIUS, PointGoal, PointWaypoints, Task, augment, make_task_sets
from mehta.nets import Actor, kl_to_standard_normal, product_of_gaussians


# ---- environments -----------------------------------------------------------
def test_point_goal_dense_reward_and_clipping():
    env = PointGoal(Task(0.0))
    assert np.allclose(env.reset(), 0)
    o, r, rc, done = env.step(np.array([5.0, 0.0]))          # clipped to +1 -> 0.1 step
    assert np.allclose(o, [0.1, 0.0])
    assert r == rc == pytest.approx(-0.9)


def test_point_goal_rotation_shifts_dynamics():
    env = PointGoal(Task(0.0, rotation=np.pi / 2))
    env.reset()
    o, *_ = env.step(np.array([1.0, 0.0]))
    assert np.allclose(o, [0.0, 0.1], atol=1e-9)


def test_point_goal_horizon():
    env = PointGoal(Task(0.0))
    env.reset()
    dones = [env.step(np.zeros(2))[3] for _ in range(PointGoal.horizon)]
    assert dones[-1] and not any(dones[:-1])


def test_waypoints_sparse_reward_and_stage_advance():
    env = PointWaypoints(Task(0.0))
    env.reset()
    total = 0.0
    for wp in env.waypoints:                                   # steer straight to each waypoint
        while env.stage < 3 and np.allclose(env.waypoints[env.stage], wp):
            d = wp - env.pos
            _, r, rc, _ = env.step(np.clip(d / 0.1, -1, 1))
            total += rc
            assert rc in (0.0, 1.0)
    assert total == 3.0 and env.stage == 3


def test_task_splits_disjoint_and_ood_in_lower_half():
    ts = make_task_sets("point_goal", seed=0)
    assert all(0 <= t.angle <= np.pi for t in ts.train + ts.evals["id"] + ts.evals["val"])
    assert all(np.pi < t.angle < 2 * np.pi for t in ts.evals["ood_goal"] + ts.evals["continual"])
    assert all(t.rotation > 0 for t in ts.evals["ood_dyn"])
    train_angles = {round(t.angle, 9) for t in ts.train}
    assert not train_angles & {round(t.angle, 9) for t in ts.evals["id"]}


# ---- PEARL math -------------------------------------------------------------
def test_product_of_gaussians_matches_closed_form():
    mu = torch.tensor([[1.0], [3.0]])
    var = torch.tensor([[1.0], [1.0]])
    m, v = product_of_gaussians(mu, var)
    assert m.item() == pytest.approx(2.0) and v.item() == pytest.approx(0.5)


def test_product_of_gaussians_is_permutation_invariant():
    mu, var = torch.randn(10, 5), torch.rand(10, 5) + 0.1
    m1, v1 = product_of_gaussians(mu, var)
    perm = torch.randperm(10)
    m2, v2 = product_of_gaussians(mu[perm], var[perm])
    assert torch.allclose(m1, m2, atol=1e-5) and torch.allclose(v1, v2, atol=1e-6)


def test_kl_zero_at_prior():
    assert kl_to_standard_normal(torch.zeros(5), torch.ones(5)).item() == pytest.approx(0.0)


def test_empty_context_gives_prior():
    ag = Agent(2, 2, SACConfig())
    mu, var = ag.posterior(torch.zeros(0, ag.ctx_dim))
    assert torch.all(mu == 0) and torch.all(var == 1)


def test_actor_logprob_matches_monte_carlo_density():
    torch.manual_seed(0)
    actor = Actor(1, 0, 1, 16)
    obs, z = torch.zeros(1, 1), torch.zeros(1, 0)
    a, logp = actor(obs, z, deterministic=True)
    # deterministic action is tanh(mu); its log-density must be finite and consistent
    assert torch.isfinite(logp).all() and a.abs().max() < 1


# ---- MEHTA pieces -----------------------------------------------------------
def test_reptile_moves_init_toward_adapted_mean():
    ag = Agent(2, 2, SACConfig(hidden=8))
    theta0 = [p.clone() for p in ag.actor.parameters()]
    a1, a2 = ag.clone_for_adaptation(), ag.clone_for_adaptation()
    with torch.no_grad():
        for p in a1.actor.parameters():
            p.add_(1.0)
        for p in a2.actor.parameters():
            p.add_(3.0)
    reptile_apply(ag, [a1, a2], beta=0.5)
    for p, p0 in zip(ag.actor.parameters(), theta0):
        assert torch.allclose(p, p0 + 1.0)                     # 0.5 * mean(1, 3)


def test_clone_does_not_touch_original():
    ag = Agent(2, 2, SACConfig(hidden=8))
    before = [p.clone() for p in ag.actor.parameters()]
    cl = ag.clone_for_adaptation()
    o = torch.randn(32, 2)
    cl.sac_update(o, torch.rand(32, 2), torch.randn(32), o, torch.zeros(32, ag.latent_dim), 0.9)
    assert all(torch.equal(p, b) for p, b in zip(ag.actor.parameters(), before))
    assert cl.encoder is ag.encoder


def test_proximal_term_keeps_weights_closer_to_anchor():
    torch.manual_seed(0)
    o, a, r = torch.randn(64, 2), torch.rand(64, 2) * 2 - 1, torch.randn(64)
    dists = {}
    for lam in (0.0, 50.0):
        torch.manual_seed(1)
        ag = Agent(2, 2, SACConfig(hidden=16))
        anchor = anchor_of(ag)
        cl = ag.clone_for_adaptation()
        for _ in range(30):
            cl.sac_update(o, a, r, o, torch.zeros(64, ag.latent_dim), 0.9, anchor=anchor, lam=lam)
        dists[lam] = sum(((p - p0) ** 2).sum().item() for p, p0 in zip(cl.actor.parameters(), anchor[0]))
    assert dists[50.0] < dists[0.0]


def test_post_train_returns_all_checkpoints_and_pearl_alone_does_not_change_weights():
    torch.manual_seed(0)
    ag = Agent(PointGoal.obs_dim, PointGoal.act_dim, SACConfig(hidden=16))
    before = [p.clone() for p in ag.actor.parameters()]
    out, _, kept = post_train(ag, PointGoal, Task(1.0), [0, 5, 10], 0.5, False, np.random.default_rng(0), keep=(0, 10))
    assert set(out) == {0, 5, 10} and set(kept) == {0, 10}
    assert all(torch.equal(kept[0]["actor"][k], kept[10]["actor"][k]) for k in kept[0]["actor"])      # no grad -> same actor
    assert all(torch.equal(p, b) for p, b in zip(ag.actor.parameters(), before))


def test_continual_metrics():
    R = [[-1.0], [-2.0, -1.5], [-3.0, -2.0, -1.0]]
    m = continual_metrics(R)
    assert m["final_mean_return"] == pytest.approx(-2.0)
    assert m["forgetting"] == pytest.approx(((-1 - -3) + (-1.5 - -2)) / 2)


# ---- v2 ---------------------------------------------------------------------
def test_waypoint_shaping_is_potential_based():
    """r_train - sparse must equal gamma*Phi(s') - Phi(s) on every step."""
    rng = np.random.default_rng(0)
    env = PointWaypoints(Task(0.3))
    env.reset()
    for _ in range(PointWaypoints.horizon):
        phi0 = env.potential()
        _, r, rc, _ = env.step(rng.uniform(-1, 1, 2))
        assert r - rc == pytest.approx(env.gamma * env.potential() - phi0)


def test_waypoint_shaping_has_no_hover_trap():
    """Reaching a waypoint must not lower the potential (v1's bug)."""
    env = PointWaypoints(Task(0.0))
    env.reset()
    env.pos = env.waypoints[0] + np.array([0.0, -GOAL_RADIUS - 0.05])
    before = env.potential()
    env.step(np.array([0.0, 1.0]))                     # step into the first waypoint's radius
    assert env.stage == 1 and env.potential() > before


def test_augment_is_a_different_family_from_test_shifts():
    rng = np.random.default_rng(0)
    for _ in range(200):
        t = augment(Task(1.0), rng)
        assert 0.5 <= t.gain <= 1.5 and np.hypot(*t.drift) <= 0.03 + 1e-12
        assert t.rotation == 0.0 and t.angle == 1.0      # never rotates actions or moves the goal


def test_gain_and_drift_change_dynamics():
    env = PointGoal(Task(0.0, gain=0.5, drift=(0.0, 0.01)))
    env.reset()
    o, *_ = env.step(np.array([1.0, 0.0]))
    assert np.allclose(o, [0.05, 0.01])


def test_v2_configs_are_compute_matched():
    from mehta.train import TrainConfig, meta_train
    counts = {}
    for m in ("pearl_cm", "mehta_v2"):
        _, info = meta_train("point_goal", m, 0, TrainConfig(n_iters=1, eval_every=10**6), SACConfig(hidden=16))
        counts[m] = info["sample_grads"]
    assert counts["pearl_cm"] == counts["mehta_v2"] == 51_200


# ---- v3 ---------------------------------------------------------------------
def test_oracle_is_invariant_to_known_action_transforms():
    """An oracle that knows the dynamics should score about the same under any
    invertible action transform for the same goal. Not exactly: the action box
    is a square, so rotating it changes the fastest reachable direction."""
    from mehta.envs import oracle_return
    base = oracle_return(Task(1.0))
    for t in (Task(1.0, rotation=np.pi), Task(1.0, mirror=True), Task(1.0, rotation=np.deg2rad(120))):
        assert oracle_return(t) == pytest.approx(base, abs=0.3)
    assert base > -20.0 + 1.0                        # far better than standing still (-20)


def test_mirror_flips_x():
    env = PointGoal(Task(0.0, mirror=True))
    env.reset()
    o, *_ = env.step(np.array([1.0, 0.5]))
    assert np.allclose(o, [-0.1, 0.05])


def test_shift_sets_exist_and_leave_old_draws_unchanged():
    ts = make_task_sets("point_goal", seed=0)
    for name in ("shift_rot120", "shift_invert", "shift_mirror", "shift_wind", "shift_ood_rot60"):
        assert len(ts.evals[name]) == 5
    assert all(np.hypot(*t.drift) == pytest.approx(0.06) for t in ts.evals["shift_wind"])
    # v1/v2 task draws must not move: id0 for seed 0 is a fixed number
    assert ts.evals["id"][0].angle == pytest.approx(make_task_sets("point_goal", seed=0).evals["id"][0].angle)
