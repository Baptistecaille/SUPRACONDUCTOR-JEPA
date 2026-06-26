"""
diffusion.py — Générateur DDPM de candidats cristallins compatibles SUPRA-JEPA.

Le modèle apprend à débruiter une représentation vectorielle continue d'un
batch cristallin tokenisé :
  - global : groupe d'espace + paramètres de maille normalisés
  - sites  : élément, Wyckoff, coordonnées fractionnelles, présence du site

Après sampling, le vecteur est discrétisé en tenseurs directement consommables
par SupraJEPA.forward_classify. Cette première version génère des candidats en
espace token ; une validation cristallographique stricte peut être ajoutée en
post-traitement avant synthèse expérimentale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


@dataclass(frozen=True)
class DiffusionShape:
    n_global: int
    n_site_features: int
    n_sites: int

    @property
    def dim(self) -> int:
        return self.n_global + self.n_sites * self.n_site_features


def diffusion_shape(cfg: Config) -> DiffusionShape:
    return DiffusionShape(n_global=7, n_site_features=6, n_sites=cfg.max_atoms)


def batch_to_diffusion_vector(batch: dict, cfg: Config) -> torch.Tensor:
    """Convertit un batch tokenisé JEPA en vecteur continu pour le DDPM."""
    element = batch["element_ids"][:, 1:].float() / (cfg.n_elements - 1)
    wyckoff = batch["wyckoff_ids"][:, 1:].float() / (cfg.n_wyckoff - 1)
    coords = batch["frac_coords"][:, 1:].float()
    valid = batch["padding_mask"][:, 1:].float()

    sites = torch.cat(
        [
            element.unsqueeze(-1),
            wyckoff.unsqueeze(-1),
            coords,
            valid.unsqueeze(-1),
        ],
        dim=-1,
    )
    global_feat = torch.cat(
        [
            batch["sg_id"].float().unsqueeze(-1) / (cfg.n_space_groups - 1),
            batch["lattice_feat"].float(),
        ],
        dim=-1,
    )
    return torch.cat([global_feat, sites.flatten(1)], dim=-1)


def diffusion_vector_to_batch(
    vector: torch.Tensor,
    cfg: Config,
    min_atoms: int = 1,
) -> dict:
    """Discrétise des samples DDPM en batch compatible avec SupraJEPA."""
    shape = diffusion_shape(cfg)
    if vector.shape[-1] != shape.dim:
        raise ValueError(f"Expected diffusion dim {shape.dim}, got {vector.shape[-1]}")

    B = vector.shape[0]
    device = vector.device

    global_feat = vector[:, : shape.n_global]
    site_feat = vector[:, shape.n_global :].reshape(B, shape.n_sites, shape.n_site_features)

    sg_id = torch.round(global_feat[:, 0].clamp(1 / 230, 1.0) * (cfg.n_space_groups - 1)).long()
    sg_id = sg_id.clamp(1, cfg.n_space_groups - 1)

    lattice_feat = global_feat[:, 1:].clone()
    lattice_feat[:, :3] = lattice_feat[:, :3].clamp(0.05, 2.5)
    lattice_feat[:, 3:] = lattice_feat[:, 3:].clamp(0.15, 1.0)

    presence_score = site_feat[:, :, 5]
    padding_atoms = presence_score > 0.5
    if min_atoms > 0:
        topk = torch.topk(presence_score, k=min(min_atoms, cfg.max_atoms), dim=1).indices
        padding_atoms.scatter_(1, topk, True)

    element_atoms = torch.round(site_feat[:, :, 0].clamp(1 / 118, 1.0) * (cfg.n_elements - 1)).long()
    wyckoff_atoms = torch.round(site_feat[:, :, 1].clamp(1 / 26, 1.0) * (cfg.n_wyckoff - 1)).long()
    element_atoms = element_atoms.clamp(1, cfg.n_elements - 1)
    wyckoff_atoms = wyckoff_atoms.clamp(1, cfg.n_wyckoff - 1)
    coords_atoms = site_feat[:, :, 2:5].clamp(0.0, 1.0)

    element_ids = torch.zeros(B, cfg.max_atoms + 1, dtype=torch.long, device=device)
    wyckoff_ids = torch.zeros(B, cfg.max_atoms + 1, dtype=torch.long, device=device)
    frac_coords = torch.zeros(B, cfg.max_atoms + 1, 3, dtype=vector.dtype, device=device)
    padding_mask = torch.zeros(B, cfg.max_atoms + 1, dtype=torch.bool, device=device)
    mask = torch.zeros_like(padding_mask)

    element_ids[:, 1:] = torch.where(padding_atoms, element_atoms, torch.zeros_like(element_atoms))
    wyckoff_ids[:, 1:] = torch.where(padding_atoms, wyckoff_atoms, torch.zeros_like(wyckoff_atoms))
    frac_coords[:, 1:] = torch.where(padding_atoms.unsqueeze(-1), coords_atoms, torch.zeros_like(coords_atoms))
    padding_mask[:, 0] = True
    padding_mask[:, 1:] = padding_atoms

    return {
        "element_ids": element_ids,
        "wyckoff_ids": wyckoff_ids,
        "frac_coords": frac_coords,
        "lattice_feat": lattice_feat,
        "sg_id": sg_id,
        "padding_mask": padding_mask,
        "mask": mask,
    }


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class CrystalDiffusionDenoiser(nn.Module):
    """MLP de débruitage epsilon_theta(x_t, t)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        dim = diffusion_shape(cfg).dim

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.diffusion_time_dim, cfg.diffusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.diffusion_hidden_dim, cfg.diffusion_hidden_dim),
        )

        layers: list[nn.Module] = []
        in_dim = dim + cfg.diffusion_hidden_dim
        for _ in range(cfg.diffusion_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, cfg.diffusion_hidden_dim),
                    nn.LayerNorm(cfg.diffusion_hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(cfg.diffusion_dropout),
                ]
            )
            in_dim = cfg.diffusion_hidden_dim
        layers.append(nn.Linear(cfg.diffusion_hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(timesteps, self.cfg.diffusion_time_dim)
        t_emb = self.time_mlp(t_emb)
        return self.net(torch.cat([x_t, t_emb], dim=-1))


class CrystalDDPM(nn.Module):
    """DDPM standard avec schedule linéaire, adapté aux tokens cristallins."""

    def __init__(
        self,
        cfg: Config,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        self.cfg = cfg
        self.denoiser = CrystalDiffusionDenoiser(cfg)

        betas = torch.linspace(beta_start, beta_end, cfg.diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_var = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_var.clamp(min=1e-20))

    def q_sample(self, x_0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def loss(self, x_0: torch.Tensor) -> torch.Tensor:
        B = x_0.shape[0]
        timesteps = torch.randint(0, self.cfg.diffusion_steps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, timesteps, noise)
        pred_noise = self.denoiser(x_t, timesteps)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        beta_t = self.betas[timesteps].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].unsqueeze(-1)
        sqrt_recip_alpha = self.sqrt_recip_alphas[timesteps].unsqueeze(-1)

        pred_noise = self.denoiser(x_t, timesteps)
        mean = sqrt_recip_alpha * (x_t - beta_t * pred_noise / sqrt_one_minus)
        noise = torch.randn_like(x_t)
        nonzero = (timesteps != 0).float().unsqueeze(-1)
        var = self.posterior_variance[timesteps].unsqueeze(-1)
        return mean + nonzero * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        self.eval()
        x = torch.randn(n_samples, diffusion_shape(self.cfg).dim, device=device)
        for step in reversed(range(self.cfg.diffusion_steps)):
            t = torch.full((n_samples,), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x
