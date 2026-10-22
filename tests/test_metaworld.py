import numpy as np
import pytest
import torch

pytest.importorskip("metaworld")

from mehta.adapt import post_train
from mehta.agent import Agent, SACConfig
from mehta.envs import ENVS, make_task_sets
from mehta.mw_envs import MW_ENVS


@pytest.mark.parametrize("name", list(MW_ENVS))
def test_metaworld_interface_and_horizon(name):
    Env = ENVS[name]
    sets = make_task_sets(name, seed=0)
    assert len(sets.train) == 40 and set(sets.evals) == {"id", "val"}
    env = Env(sets.evals["id"][0])
    o = env.reset()
    assert o.shape == (Env.obs_dim,) and np.allclose(o[-3:], 0)       # goal hidden from the agent
    dones = []
    for _ in range(Env.horizon):
        o, r, rc, done = env.step(np.zeros(Env.act_dim))
        assert r == rc and 0.0 <= r <= 1.0
        dones.append(done)
    assert dones[-1] and not any(dones[:-1])


def test_metaworld_splits_disjoint_and_seeded():
    a, b = make_task_sets("mw_push", seed=0), make_task_sets("mw_push", seed=1)
    train = {t.as_dict()["goal"].__repr__() for t in a.train}
    assert not train & {t.as_dict()["goal"].__repr__() for t in a.evals["id"]}
    assert [t.index for t in a.evals["id"]] != [t.index for t in b.evals["id"]]


def test_metaworld_shared_sim_switches_tasks():
    Env = ENVS["mw_reach"]
    t0, t1 = make_task_sets("mw_reach").train[:2]
    e0, e1 = Env(t0), Env(t1)
    e0.reset(); e1.reset()
    r1 = e1.step(np.zeros(4))[1]
    e0.reset()                                          # must swap the shared sim back to t0
    r0 = e0.step(np.zeros(4))[1]
    assert e0.sim is e1.sim and r0 != r1


def test_metaworld_post_train_runs():
    torch.manual_seed(0)
    Env = ENVS["mw_reach"]
    agent = Agent(Env.obs_dim, Env.act_dim, SACConfig())
    task = make_task_sets("mw_reach").evals["id"][0]
    out, _, _ = post_train(agent, Env, task, [0, 2], 0.5, True, np.random.default_rng(0))
    assert set(out) == {0, 2} and all(np.isfinite(v) for v in out.values())
