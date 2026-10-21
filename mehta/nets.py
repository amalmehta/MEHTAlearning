"""Networks: tanh-Gaussian actor, Q-function, and PEARL's product-of-Gaussians encoder."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


def mlp(inp: int, out: int, hidden: int = 128, depth: int = 2) -> nn.Sequential:
    layers, d = [], inp
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(obs_dim + latent_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False):
        h = self.net(torch.cat([obs, z], -1))
        mu, log_std = h[..., : self.act_dim], h[..., self.act_dim :]
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        # log N(u; mu, std) minus the tanh change-of-variables term (numerically stable form)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class QNet(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(obs_dim + latent_dim + act_dim, 1, hidden)

    def forward(self, obs, z, act):
        return self.net(torch.cat([obs, z, act], -1)).squeeze(-1)


class ContextEncoder(nn.Module):
    """PEARL encoder: each transition c_n gives a Gaussian factor N(mu_n, var_n);
    the posterior over z is their normalized product (permutation invariant)."""

    def __init__(self, ctx_dim: int, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = mlp(ctx_dim, 2 * latent_dim, hidden)
        self.latent_dim = latent_dim

    def posterior(self, ctx: torch.Tensor):
        """ctx: [..., N, ctx_dim] -> (mu, var) each [..., latent_dim]."""
        h = self.net(ctx)
        mu_n = h[..., : self.latent_dim]
        var_n = F.softplus(h[..., self.latent_dim :]).clamp(min=1e-7)
        return product_of_gaussians(mu_n, var_n)


def product_of_gaussians(mu_n: torch.Tensor, var_n: torch.Tensor):
    """Product of N diagonal Gaussians along dim -2: precision-weighted mean."""
    precision = (1.0 / var_n).sum(-2)
    var = 1.0 / precision
    mu = var * (mu_n / var_n).sum(-2)
    return mu, var


def kl_to_standard_normal(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return 0.5 * (var + mu**2 - 1.0 - var.log()).sum(-1)
