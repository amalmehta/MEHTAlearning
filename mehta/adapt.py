"""Post-training (test-time adaptation) on a new task, and the continual sequence.

Every variant follows the same protocol so only one thing changes at a time:
  1. explore N_EXPLORE episodes (z from prior, then posterior samples)
  2. for k = 1..K: every COLLECT_EVERY steps collect one more episode
     (posterior-sampled z); if grad=True take one SAC step on the task's own
     data using only the test-time reward, with proximal weight lam toward the
     meta-init
  3. at each checkpoint K, score one deterministic episode with z = posterior
     mean of ALL context collected so far
With grad=False (PEARL alone) step 2 still collects context, so PEARL gets the
same amount of data; it just takes no gradient steps.
"""
from __future__ import annotations

import numpy as np
import torch

from .agent import Agent, anchor_of
from .buffers import TaskBuffer, to_context
from .rollout import rollout, sample_z

N_EXPLORE = 2
COLLECT_EVERY = 10
BATCH = 128


class TaskState:
    """Everything one test task accumulates: replay data and context."""

    def __init__(self, Env, task, agent: Agent):
        self.env = Env(task)
        self.buf = TaskBuffer(Env.obs_dim, Env.act_dim, 20_000)
        self.gamma = Env.gamma
        for _ in range(N_EXPLORE):
            self.collect(agent)

    def ctx(self):
        return self.buf.context()

    def collect(self, agent):
        p = rollout(self.env, agent, sample_z(agent, self.ctx()))
        self.buf.add_path(p)

    @torch.no_grad()
    def score(self, agent, record=False):
        p = rollout(self.env, agent, sample_z(agent, self.ctx(), mean=True), deterministic=True, record=record)
        return (p["ret"], p) if record else p["ret"]


def post_train(agent: Agent, Env, task, checkpoints, lam: float, grad: bool, rng, record=False, keep=()):
    """Returns {K: return}; with record, also {K: xy-trajectory}; `keep` lists
    checkpoints whose adapted actor weights should be returned (for the demo)."""
    learner = agent.clone_for_adaptation()
    anchor = anchor_of(agent)
    st = TaskState(Env, task, learner)
    out, trajs, kept = {}, {}, {}
    kmax = max(checkpoints)
    for k in range(kmax + 1):
        if k in checkpoints:
            if record:
                out[k], p = st.score(learner, record=True)
                trajs[k] = p["xy"]
            else:
                out[k] = st.score(learner)
            if k in keep:
                kept[k] = {"actor": {n: t.clone() for n, t in learner.actor.state_dict().items()},
                           "z": sample_z(learner, st.ctx(), mean=True).detach().clone()}
        if k == kmax:
            break
        if k % COLLECT_EVERY == 0:
            st.collect(learner)
        if grad:
            _step(learner, st, anchor, lam, rng)
    return out, trajs, kept


def _step(learner: Agent, st: TaskState, anchor, lam, rng):
    with torch.no_grad():
        z = sample_z(learner, st.ctx(), mean=True)
    o, a, r, o2 = st.buf.sample(rng, BATCH, reward_key="rc")
    learner.sac_update(o, a, r, o2, z.expand(len(o), -1), st.gamma, anchor=anchor, lam=lam)


def continual(agent: Agent, Env, tasks, k_per_task: int, lam: float, grad: bool, rng):
    """Post-train on tasks in sequence, carrying weights over. Anchor = meta-init.
    Returns R where R[i][j] = return on task j after finishing task i (j <= i)."""
    learner = agent.clone_for_adaptation()
    anchor = anchor_of(agent)
    states, R = [], []
    for i, task in enumerate(tasks):
        st = TaskState(Env, task, learner)
        states.append(st)
        for k in range(k_per_task):
            if k % COLLECT_EVERY == 0:
                st.collect(learner)
            if grad:
                _step(learner, st, anchor, lam, rng)
        R.append([states[j].score(learner) for j in range(i + 1)])
    return R


def continual_metrics(R):
    n = len(R)
    final = float(np.mean(R[-1]))
    forgetting = float(np.mean([R[i][i] - R[-1][i] for i in range(n - 1)])) if n > 1 else 0.0
    return {"final_mean_return": final, "forgetting": forgetting}
