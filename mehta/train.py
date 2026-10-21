"""Meta-training for every method. One loop, three switches:

  method        latent (PEARL)   long-horizon first-order MAML step
  pearl         yes              no
  mehta         yes              yes, K_inner = 50
  mehta_short   yes              yes, K_inner = 5     (ablation: short horizon)
  maml          no               yes, K_inner = 50    (MAML-family baseline)

Version 2 (after the v1 pilot showed no gain from the MAML step):
  pearl_cm      PEARL with 100 updates/iteration: compute-matched to mehta_v2
  pearl_dr_cm   pearl_cm + half its training episodes on perturbed dynamics
                (control: is any gain just from seeing perturbed data?)
  mehta_v2      PEARL (75 updates/iter) + a MAML step whose inner loop RUNS THE
                TEST-TIME PROTOCOL (fresh episodes, online collection, test-time
                reward, proximal pull) on perturbed training tasks, every iteration
Physics domain randomization (randomize.py): pearl_phys, mehta_phys and
maml_phys are pearl_cm, mehta_v2 and maml with EVERY training episode (and
mehta_phys's inner loop) on physics drawn from the training range (mass,
friction, actuator gain, observation noise, action delay) instead of augment().
Sample-gradients per iteration: pearl_cm 100x4x128 = 51,200;
mehta_v2 75x4x128 + 2 tasks x 50 steps x 128 = 51,200.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
import torch

from .agent import Agent, SACConfig, anchor_of, reptile_apply
from .buffers import TaskBuffer
from .envs import ENVS, augment, make_task_sets
from .randomize import Randomized
from .rollout import rollout, sample_z
from .adapt import BATCH, COLLECT_EVERY, TaskState, _step as adapt_step

log = logging.getLogger("mehta")

METHODS = {
    "pearl": dict(use_latent=True, k_inner=0),
    "mehta": dict(use_latent=True, k_inner=50),
    "mehta_short": dict(use_latent=True, k_inner=5),
    "maml": dict(use_latent=False, k_inner=50),
    "pearl_cm": dict(use_latent=True, k_inner=0, tc=dict(grad_steps=100)),
    "pearl_dr_cm": dict(use_latent=True, k_inner=0, dr=0.5, tc=dict(grad_steps=100)),
    "mehta_v2": dict(use_latent=True, k_inner=50, online=True, tc=dict(reptile_every=1, beta=0.5)),
    "pearl_phys": dict(use_latent=True, k_inner=0, phys=True, tc=dict(grad_steps=100)),
    "mehta_phys": dict(use_latent=True, k_inner=50, online=True, phys=True, tc=dict(reptile_every=1, beta=0.5)),
    "maml_phys": dict(use_latent=False, k_inner=50, phys=True),
}


@dataclass
class TrainConfig:
    n_iters: int = 300
    init_eps: int = 2              # prior episodes per train task before training
    tasks_per_iter: int = 5
    prior_eps: int = 1
    post_eps: int = 2
    grad_steps: int = 75           # PEARL updates per iteration
    meta_batch: int = 4
    batch: int = 128
    ctx_batch: int = 64
    reptile_every: int = 2
    reptile_tasks: int = 2
    beta: float = 0.3              # outer (meta) step size
    inner_lam: float = 0.5         # proximal weight inside the meta-training inner loop
    eval_every: int = 20


def meta_train(env_name: str, method: str, seed: int, tc: TrainConfig, sc: SACConfig):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Env = ENVS[env_name]
    tasks = make_task_sets(env_name, seed=seed)
    m = METHODS[method]
    tc = replace(tc, **m.get("tc", {}))
    agent = Agent(Env.obs_dim, Env.act_dim, sc, use_latent=m["use_latent"])
    n = len(tasks.train)
    rl_bufs = [TaskBuffer(Env.obs_dim, Env.act_dim) for _ in range(n)]
    enc_bufs = [TaskBuffer(Env.obs_dim, Env.act_dim, 5_000) for _ in range(n)]
    prior = torch.zeros(agent.latent_dim)
    # physics DR: its own seeded stream, so it never shifts the other methods' RNG draws
    phys = Randomized(Env, seed=7_000 + seed) if m.get("phys") else None

    def collect(i, n_prior, n_post):
        task = tasks.train[i]
        if m.get("dr") and rng.random() < m["dr"]:
            task = augment(task, rng)
        env = phys(task) if phys else Env(task)
        enc_bufs[i].clear()
        for _ in range(n_prior):
            p = rollout(env, agent, prior + torch.randn(agent.latent_dim))
            rl_bufs[i].add_path(p); enc_bufs[i].add_path(p)
        for _ in range(n_post):
            p = rollout(env, agent, sample_z(agent, enc_bufs[i].context()))
            rl_bufs[i].add_path(p); enc_bufs[i].add_path(p)

    for i in range(n):
        collect(i, tc.init_eps, 0)

    history, t0, grad_count, sample_grads = [], time.time(), 0, 0
    for it in range(tc.n_iters):
        for i in rng.choice(n, tc.tasks_per_iter, replace=False):
            collect(i, tc.prior_eps, tc.post_eps)

        for _ in range(tc.grad_steps):
            idx = rng.choice(n, tc.meta_batch, replace=False)
            bs = [rl_bufs[i].sample(rng, tc.batch) for i in idx]
            o, a, r, o2 = (torch.stack(x) for x in zip(*bs))
            ctx = torch.stack([enc_bufs[i].context(rng, tc.ctx_batch) for i in idx])
            agent.pearl_update(o, a, r, o2, ctx, Env.gamma)
            grad_count += 1
            sample_grads += tc.meta_batch * tc.batch

        if m["k_inner"] and it % tc.reptile_every == 0:
            if m.get("online"):
                k = reptile_step_online(agent, Env, tasks.train, rng, m["k_inner"], tc, phys)
                grad_count += k; sample_grads += k * BATCH
            else:
                k = reptile_step(agent, rl_bufs, enc_bufs, rng, m["k_inner"], tc, Env.gamma)
                grad_count += k; sample_grads += k * tc.batch

        if it % tc.eval_every == 0 or it == tc.n_iters - 1:
            val = quick_eval(agent, Env, tasks.evals["val"])
            history.append({"iter": it, "val_return": val, "grad_steps": grad_count, "sec": round(time.time() - t0, 1)})
            log.info("%s %s seed=%d it=%d val=%.3f (%.0fs)", env_name, method, seed, it, val, time.time() - t0)
    return agent, {"history": history, "train_cfg": asdict(tc), "sac_cfg": asdict(sc), "grad_steps": grad_count,
                   "sample_grads": sample_grads}


def reptile_step_online(agent, Env, train_tasks, rng, k_inner, tc: TrainConfig, phys=None) -> int:
    """MEHTA v2's MAML step: on a perturbed training task, run exactly the
    test-time post-training protocol (explore, infer z, K_inner proximal SAC steps
    on freshly collected data with the test-time reward), then move the init
    toward the mean adapted weights. The latent cannot explain the perturbation,
    so the gradient steps have a residual to learn to fix. With `phys`, the
    perturbation is a physics draw from the training range instead."""
    anchor = anchor_of(agent)
    adapted = []
    for i in rng.choice(len(train_tasks), tc.reptile_tasks, replace=False):
        learner = agent.clone_for_adaptation()
        task = train_tasks[i]
        st = TaskState(phys, task, learner) if phys else TaskState(Env, augment(task, rng), learner)
        for k in range(k_inner):
            if k % COLLECT_EVERY == 0:
                st.collect(learner)
            adapt_step(learner, st, anchor, tc.inner_lam, rng)
        adapted.append(learner)
    reptile_apply(agent, adapted, tc.beta)
    return k_inner * tc.reptile_tasks


def reptile_step(agent, rl_bufs, enc_bufs, rng, k_inner, tc: TrainConfig, gamma) -> int:
    """Long-horizon first-order MAML: K_inner proximal SAC steps per task from the
    current init, then move the init toward the mean of the adapted weights."""
    anchor = anchor_of(agent)
    adapted = []
    for i in rng.choice(len(rl_bufs), tc.reptile_tasks, replace=False):
        learner = agent.clone_for_adaptation()
        with torch.no_grad():
            z = sample_z(agent, enc_bufs[i].context(rng, tc.ctx_batch), mean=True)
        for _ in range(k_inner):
            # train as you test: the inner loop sees only the reward available at test time
            o, a, r, o2 = rl_bufs[i].sample(rng, tc.batch, reward_key="rc")
            learner.sac_update(o, a, r, o2, z.expand(len(o), -1), gamma, anchor=anchor, lam=tc.inner_lam)
        adapted.append(learner)
    reptile_apply(agent, adapted, tc.beta)
    return k_inner * tc.reptile_tasks


@torch.no_grad()
def quick_eval(agent, Env, tasks, n_ctx_eps: int = 2) -> float:
    """PEARL-style: explore n_ctx_eps episodes (prior, then posterior samples),
    then score one deterministic episode with the posterior mean."""
    rets = []
    for task in tasks:
        env = Env(task)
        ctx = torch.zeros(0, agent.ctx_dim)
        for _ in range(n_ctx_eps):
            p = rollout(env, agent, sample_z(agent, ctx))
            ctx = torch.cat([ctx, _ctx(p)])
        rets.append(rollout(env, agent, sample_z(agent, ctx, mean=True), deterministic=True)["ret"])
    return float(np.mean(rets))


def _ctx(p):
    from .buffers import to_context
    return to_context(p["o"], p["a"], p["rc"], p["o2"])
