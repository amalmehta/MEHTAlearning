"""One training run + all of its evaluations -> results/<env>/<method>_s<seed>.json

  uv run python -m mehta.experiments --env point_goal --method mehta --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .adapt import continual, continual_metrics, post_train
from .agent import SACConfig
from .envs import ENVS, make_task_sets
from .train import TrainConfig, meta_train

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
LAM = 0.5                          # proximal weight at post-training (chosen on dev seed)
CKPT = {"point_goal": [0, 25, 50, 100, 200, 400], "waypoints": [0, 50, 100, 200, 400]}
AXIS_A_SETS = {"point_goal": ["id", "ood_goal", "ood_dyn"], "waypoints": ["id"]}
K_CONTINUAL = 100
DEMO_KEEP = (0, 100, 400)

# which evaluation variants each trained model produces: name -> (grad, lam)
VARIANTS = {
    "pearl": {"pearl": (False, 0.0), "pearl_ft": (True, LAM)},
    "mehta": {"mehta": (True, LAM)},
    "mehta_short": {"mehta_short": (True, LAM)},
    "maml": {"maml": (True, LAM)},
    "pearl_cm": {"pearl_cm": (False, 0.0), "pearl_cm_ft": (True, LAM)},
    "pearl_dr_cm": {"pearl_dr_cm": (False, 0.0), "pearl_dr_cm_ft": (True, LAM)},
    "mehta_v2": {"mehta_v2": (True, LAM)},
}
CONTINUAL_VARIANTS = {
    "pearl": {"pearl": (False, 0.0), "pearl_ft": (True, LAM)},
    "mehta": {"mehta": (True, LAM), "mehta_lam0": (True, 0.0)},
    "maml": {"maml": (True, LAM)},
    "pearl_cm": {"pearl_cm_ft": (True, LAM)},
    "pearl_dr_cm": {"pearl_dr_cm_ft": (True, LAM)},
    "mehta_v2": {"mehta_v2": (True, LAM), "mehta_v2_lam0": (True, 0.0)},
}
RECORD_SEEDS = {0, 3}               # seed 0: v1 demo; seed 3: first v2 confirmatory seed


def run(env_name: str, method: str, seed: int, sets=None, out_dir: Path = RESULTS) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    Env = ENVS[env_name]
    agent, info = meta_train(env_name, method, seed, TrainConfig(), SACConfig())
    tasks = make_task_sets(env_name, seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    record = seed in RECORD_SEEDS
    res = {"env": env_name, "method": method, "seed": seed, "lam": LAM, "train": info, "axis_a": {}, "traj": {}}
    models = out_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    for vname, (grad, lam) in VARIANTS[method].items():
        res["axis_a"][vname] = {}
        for set_name in sets or AXIS_A_SETS[env_name]:
            per_task, trajs = [], []
            for ti, task in enumerate(tasks.evals[set_name]):
                keep = DEMO_KEEP if (record and set_name == "ood_goal") else ()
                out, tr, kept = post_train(agent, Env, task, CKPT[env_name], lam, grad, rng, record=record, keep=keep)
                per_task.append({str(k): v for k, v in out.items()})
                trajs.append({"task": task.as_dict(), "xy": {str(k): v for k, v in tr.items()}})
                if kept:
                    torch.save(kept, models / f"{env_name}_{vname}_s{seed}_{set_name}{ti}_adapted.pt")
            res["axis_a"][vname][set_name] = per_task
            if record:
                res["traj"].setdefault(vname, {})[set_name] = trajs
        logging.getLogger("mehta").info("%s %s seed=%d evaluated %s (%.0fs)", env_name, method, seed, vname, time.time() - t0)

    if env_name == "point_goal" and method in CONTINUAL_VARIANTS:
        res["axis_c"] = {}
        for vname, (grad, lam) in CONTINUAL_VARIANTS[method].items():
            R = continual(agent, Env, tasks.evals["continual"], K_CONTINUAL, lam, grad, rng)
            res["axis_c"][vname] = {"R": R, **continual_metrics(R)}

    if record:
        torch.save(
            {"actor": agent.actor.state_dict(), "encoder": agent.encoder.state_dict() if agent.encoder else None,
             "latent_dim": agent.latent_dim, "hidden": agent.cfg.hidden},
            models / f"{env_name}_{method}_s{seed}.pt",
        )
    res["wall_sec"] = round(time.time() - t0, 1)
    path = out_dir / env_name / f"{method}_s{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=list(ENVS))
    ap.add_argument("--method", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(a.env, a.method, a.seed, out_dir=Path(a.out))


if __name__ == "__main__":
    main()
