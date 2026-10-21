"""Why did v1's long-horizon MAML step not help? Re-runs v1 MEHTA meta-training
(Point-Goal, same config) with instrumentation on every MAML step:

  inner_gain     return on the inner-loop task after K_inner steps minus before
                 (deterministic episode, z = posterior mean of that task's context)
  step_ratio     ||beta * mean(theta_i - theta)|| / ||change from the PEARL updates
                 since the previous MAML step||
  cos            cosine similarity between the two tasks' inner-loop updates

  uv run python -m mehta.diagnose_v1 [--seed 0] [--iters 300]
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch

from . import train as T
from .agent import SACConfig, anchor_of, reptile_apply
from .envs import ENVS, make_task_sets
from .experiments import RESULTS
from .rollout import rollout, sample_z

log = logging.getLogger("mehta")


def flat(agent) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for m in agent.policy_modules() for p in m.parameters()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(1)
    Env = ENVS["point_goal"]
    train_tasks = make_task_sets("point_goal", seed=a.seed).train
    records, state = [], {"prev": None}

    @torch.no_grad()
    def score(agent, task, ctx):
        z = sample_z(agent, ctx, mean=True)
        return rollout(Env(task), agent, z, deterministic=True)["ret"]

    def instrumented(agent, rl_bufs, enc_bufs, rng, k_inner, tc, gamma):
        before = flat(agent)
        anchor = anchor_of(agent)
        adapted, gains, deltas = [], [], []
        for i in rng.choice(len(rl_bufs), tc.reptile_tasks, replace=False):
            learner = agent.clone_for_adaptation()
            ctx = enc_bufs[i].context()
            with torch.no_grad():
                z = sample_z(agent, enc_bufs[i].context(rng, tc.ctx_batch), mean=True)
            r0 = score(learner, train_tasks[i], ctx)
            for _ in range(k_inner):
                o, act, r, o2 = rl_bufs[i].sample(rng, tc.batch, reward_key="rc")
                learner.sac_update(o, act, r, o2, z.expand(len(o), -1), gamma, anchor=anchor, lam=tc.inner_lam)
            gains.append(score(learner, train_tasks[i], ctx) - r0)
            deltas.append(flat(learner) - before)
            adapted.append(learner)
        reptile_apply(agent, adapted, tc.beta)
        meta_step = (flat(agent) - before).norm().item()
        pearl_change = (before - state["prev"]).norm().item() if state["prev"] is not None else float("nan")
        state["prev"] = flat(agent)
        cos = torch.nn.functional.cosine_similarity(deltas[0], deltas[1], dim=0).item()
        records.append({"inner_gain": [round(g, 4) for g in gains], "meta_step": meta_step,
                        "pearl_change": pearl_change, "step_ratio": meta_step / pearl_change if pearl_change == pearl_change else None,
                        "cos": cos})
        return k_inner * tc.reptile_tasks

    T.reptile_step = instrumented
    T.meta_train("point_goal", "mehta", a.seed, T.TrainConfig(n_iters=a.iters), SACConfig())

    gains = np.array([g for r in records for g in r["inner_gain"]])
    ratios = np.array([r["step_ratio"] for r in records if r["step_ratio"] is not None])
    coss = np.array([r["cos"] for r in records])
    late = records[len(records) // 2:]
    summary = {
        "n_meta_steps": len(records),
        "inner_gain_mean": float(gains.mean()), "inner_gain_std": float(gains.std()),
        "inner_gain_late_mean": float(np.mean([g for r in late for g in r["inner_gain"]])),
        "step_ratio_median": float(np.median(ratios)),
        "cos_mean": float(coss.mean()), "cos_late_mean": float(np.mean([r["cos"] for r in late])),
    }
    out = RESULTS / "diagnostics" / f"v1_inner_loop_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
