"""SAC agent conditioned on a PEARL latent z, plus the pieces MEHTA adds on top:
cloning for an inner loop, a proximal anchor, and the first-order meta-update."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .nets import Actor, ContextEncoder, QNet, kl_to_standard_normal, mlp


@dataclass
class SACConfig:
    hidden: int = 64
    latent_dim: int = 5          # PEARL's default latent size
    lr: float = 3e-4
    tau: float = 0.005
    kl_lambda: float = 1e-3      # PEARL default is 0.1 at reward scale 100; dev seed: 1e-3 at scale 1
    reward_scale: float = 1.0
    init_alpha: float = 0.1      # dev-seed finding: alpha=1 lets entropy swamp a ~-1/step reward
    recon_weight: float = 0.0    # auxiliary reward decoder on the encoder; dev seed showed no gain, so off


class Agent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: SACConfig, use_latent: bool = True):
        self.cfg = cfg
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.latent_dim = cfg.latent_dim if use_latent else 0
        ld, h = self.latent_dim, cfg.hidden
        self.actor = Actor(obs_dim, ld, act_dim, h)
        self.q1, self.q2 = QNet(obs_dim, ld, act_dim, h), QNet(obs_dim, ld, act_dim, h)
        self.q1_t, self.q2_t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.ctx_dim = obs_dim + act_dim + 1 + obs_dim        # (o, a, r, o')
        self.encoder = ContextEncoder(self.ctx_dim, ld, h) if use_latent else None
        self.decoder = mlp(obs_dim + ld, 1, h) if use_latent and cfg.recon_weight > 0 else None
        self.log_alpha = torch.tensor(float(np.log(cfg.init_alpha)), requires_grad=True)
        self.target_entropy = -float(act_dim)
        self._make_optimizers(train_encoder=use_latent)

    # ----- parameters -------------------------------------------------------
    def policy_modules(self):
        """The parameters MAML meta-learns an initialization for."""
        return [self.actor, self.q1, self.q2]

    def _make_optimizers(self, train_encoder: bool):
        q_params = list(self.q1.parameters()) + list(self.q2.parameters())
        if train_encoder and self.encoder is not None:
            q_params += list(self.encoder.parameters())
            if self.decoder is not None:
                q_params += list(self.decoder.parameters())
        self.q_opt = torch.optim.Adam(q_params, lr=self.cfg.lr, foreach=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr, foreach=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr)

    def clone_for_adaptation(self) -> "Agent":
        """Copy the policy nets with fresh optimizers; share the (frozen) encoder."""
        new = copy.copy(self)
        for name in ("actor", "q1", "q2", "q1_t", "q2_t"):
            setattr(new, name, copy.deepcopy(getattr(self, name)))
        new.log_alpha = self.log_alpha.detach().clone().requires_grad_(True)
        new._make_optimizers(train_encoder=False)
        return new

    def flat_policy_params(self) -> list[torch.Tensor]:
        return [p.detach().clone() for m in self.policy_modules() for p in m.parameters()]

    # ----- latent -----------------------------------------------------------
    def posterior(self, ctx: torch.Tensor):
        """ctx [N, ctx_dim] or [T, N, ctx_dim]. Empty context -> the prior N(0, I)."""
        if self.encoder is None:
            shape = ctx.shape[:-2] + (0,)
            return torch.zeros(shape), torch.ones(shape)
        if ctx.shape[-2] == 0:
            shape = ctx.shape[:-2] + (self.latent_dim,)
            return torch.zeros(shape), torch.ones(shape)
        return self.encoder.posterior(ctx)

    @torch.no_grad()
    def act(self, obs: np.ndarray, z: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32)
        a, _ = self.actor(o, z, deterministic)
        return a.numpy()

    # ----- SAC update -------------------------------------------------------
    def sac_update(self, o, a, r, o2, z, gamma: float, kl=None, anchor=None, lam: float = 0.0, extra=None):
        """One SAC step. z carries gradient into the encoder through the critic
        (as in PEARL); the actor and the targets see z.detach(). If `anchor` is
        given, adds the proximal term lam/2 * ||theta - anchor||^2."""
        zd = z.detach()
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor(o2, zd)
            qt = torch.min(self.q1_t(o2, zd, a2), self.q2_t(o2, zd, a2)) - alpha * logp2
            y = self.cfg.reward_scale * r + gamma * qt
        q_loss = F.mse_loss(self.q1(o, z, a), y) + F.mse_loss(self.q2(o, z, a), y)
        if kl is not None:
            q_loss = q_loss + self.cfg.kl_lambda * kl
        if extra is not None:
            q_loss = q_loss + extra
        if anchor is not None and lam > 0:
            q_loss = q_loss + 0.5 * lam * _sqdist([self.q1, self.q2], anchor[1:])
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        a_pi, logp = self.actor(o, zd)
        q_pi = torch.min(self.q1(o, zd, a_pi), self.q2(o, zd, a_pi))
        actor_loss = (alpha * logp - q_pi).mean()
        if anchor is not None and lam > 0:
            actor_loss = actor_loss + 0.5 * lam * _sqdist([self.actor], anchor[:1])
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                torch._foreach_lerp_(list(tgt.parameters()), list(net.parameters()), self.cfg.tau)
        return float(q_loss), float(actor_loss)

    def pearl_update(self, o, a, r, o2, ctx, gamma: float):
        """Meta-batch update. o: [T, B, obs], ctx: [T, C, ctx_dim]."""
        T, B = o.shape[:2]
        mu, var = self.posterior(ctx)
        z = mu + var.sqrt() * torch.randn_like(mu)                 # reparameterized
        kl = kl_to_standard_normal(mu, var).sum() if self.encoder is not None else None
        zb = z[:, None, :].expand(T, B, z.shape[-1])
        f = lambda x: x.reshape(T * B, -1)
        extra = None
        if self.decoder is not None:
            r_hat = self.decoder(torch.cat([f(o2), f(zb)], -1)).squeeze(-1)
            extra = self.cfg.recon_weight * F.mse_loss(r_hat, r.reshape(-1))
        return self.sac_update(f(o), f(a), r.reshape(-1), f(o2), f(zb), gamma, kl=kl, extra=extra)

    def snapshot(self) -> dict:
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "q1": copy.deepcopy(self.q1.state_dict()),
            "q2": copy.deepcopy(self.q2.state_dict()),
        }


def _sqdist(modules, anchor_groups) -> torch.Tensor:
    total = torch.zeros(())
    for m, anchors in zip(modules, anchor_groups):
        for p, p0 in zip(m.parameters(), anchors):
            total = total + ((p - p0) ** 2).sum()
    return total


def anchor_of(agent: Agent):
    """Frozen copy of the policy params, grouped as [actor, q1, q2]."""
    return [[p.detach().clone() for p in m.parameters()] for m in agent.policy_modules()]


@torch.no_grad()
def reptile_apply(agent: Agent, adapted: list[Agent], beta: float):
    """First-order meta-update: theta <- theta + beta * mean_i(theta_i' - theta)."""
    for gi, m in enumerate(agent.policy_modules()):
        for pi, p in enumerate(m.parameters()):
            delta = torch.stack([list(a.policy_modules()[gi].parameters())[pi] - p for a in adapted]).mean(0)
            p.add_(beta * delta)
