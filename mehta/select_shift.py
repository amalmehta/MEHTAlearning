"""Pick v3's held-out shift on the dev seed, using ONLY the PEARL baseline.

Rule (pre-registered in MEHTAlearning.md before this ran): fine-tune
compute-matched PEARL (pearl_cm_ft) on each candidate shift; gap closure =
(R(400) - R(0)) / (oracle - R(0)); pick the lowest closure among candidates whose
K=400 return is still at least 30% of the oracle gap short. MEHTA is never
evaluated here.

  uv run python -m mehta.select_shift [--seed 100]
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch

from .adapt import post_train
from .agent import SACConfig
from .envs import SHIFTS, PointGoal, make_task_sets, oracle_return
from .experiments import LAM, RESULTS
from .train import TrainConfig, meta_train

CKPT = [0, 100, 200, 400]
MIN_SHORTFALL = 0.30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    torch.set_num_threads(1)
    agent, _ = meta_train("point_goal", "pearl_cm", a.seed, TrainConfig(), SACConfig())
    tasks = make_task_sets("point_goal", seed=a.seed).evals
    rng = np.random.default_rng(20_000 + a.seed)
    rows = {}
    for name in SHIFTS:
        rets = [post_train(agent, PointGoal, t, CKPT, LAM, True, rng)[0] for t in tasks[name]]
        mean = {k: float(np.mean([r[k] for r in rets])) for k in CKPT}
        oracle = float(np.mean([oracle_return(t) for t in tasks[name]]))
        gap0 = oracle - mean[0]
        closure = (mean[400] - mean[0]) / gap0 if gap0 > 1e-9 else float("nan")
        shortfall = (oracle - mean[400]) / gap0 if gap0 > 1e-9 else 0.0
        rows[name] = {"return": mean, "oracle": oracle, "closure": closure, "shortfall_at_400": shortfall,
                      "eligible": bool(shortfall >= MIN_SHORTFALL)}
        logging.getLogger("mehta").info("%s: R0=%.2f R400=%.2f oracle=%.2f closure=%.2f shortfall=%.2f",
                                        name, mean[0], mean[400], oracle, closure, shortfall)
    eligible = {k: v for k, v in rows.items() if v["eligible"]}
    chosen = min(eligible, key=lambda k: eligible[k]["closure"]) if eligible else None
    out = RESULTS / "shift_selection" / f"dev_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rule": __doc__, "method": "pearl_cm_ft", "lam": LAM, "rows": rows, "chosen": chosen}, indent=1))
    print(json.dumps({"chosen": chosen, **{k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                                              for kk, vv in v.items() if kk != "return"} for k, v in rows.items()}}, indent=1))


if __name__ == "__main__":
    main()
