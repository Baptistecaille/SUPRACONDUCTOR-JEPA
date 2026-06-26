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


def _to_diffusion_range(x: torch.Tensor) -> torch.Tensor:
    """Mappe les features bornées [0, 1] vers [-1, 1] pour le DDPM."""
    return x * 2.0 - 1.0


def _from_diffusion_range(x: torch.Tensor) -> torch.Tensor:
    """Ramène les samples DDPM vers l'intervalle feature [0, 1]."""
    return ((torch.tanh(x) + 1.0) * 0.5).clamp(0.0, 1.0)


def _quantile_lookup(values: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Mappe des probabilités [0, 1] vers des valeurs empiriques triées."""
    flat_values = values.flatten().to(probs.device)
    if flat_values.numel() == 0:
        raise ValueError("Cannot decode from empty empirical metadata.")
    sorted_values = torch.sort(flat_values.float()).values
    idx = torch.round(probs.clamp(0.0, 1.0) * (sorted_values.numel() - 1)).long()
    return sorted_values[idx]


def _quantile_lookup_rows(rows: torch.Tensor, probs: torch.Tensor, sort_col: int = 0) -> torch.Tensor:
    """Sélectionne des lignes empiriques complètes en conservant leurs corrélations."""
    rows = rows.to(probs.device).float()
    if rows.shape[0] == 0:
        raise ValueError("Cannot decode from empty empirical rows.")
    order = torch.argsort(rows[:, sort_col])
    sorted_rows = rows[order]
    idx = torch.round(probs.clamp(0.0, 1.0) * (sorted_rows.shape[0] - 1)).long()
    return sorted_rows[idx]


@torch.no_grad()
def collect_diffusion_metadata(loader, cfg: Config, device: torch.device | None = None) -> dict:
    """Collecte les distributions empiriques utilisées pour décoder les samples.

    Le DDPM prédit des variables continues. Pour les champs discrets
    (éléments, Wyckoff, groupe d'espace, nombre d'atomes), on décode par
    quantiles observés dans les données d'entraînement plutôt que par un
    arrondi ordinal qui favorise artificiellement H/Og et a/z.
    """
    device = device or torch.device("cpu")
    elements, wyckoffs, space_groups, lattices, atom_counts, coords = [], [], [], [], [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        valid = batch["padding_mask"][:, 1:]
        elements.append(batch["element_ids"][:, 1:][valid].detach().cpu())
        wyckoffs.append(batch["wyckoff_ids"][:, 1:][valid].detach().cpu())
        coords.append(batch["frac_coords"][:, 1:][valid].detach().cpu())
        space_groups.append(batch["sg_id"].detach().cpu())
        lattices.append(batch["lattice_feat"].detach().cpu())
        atom_counts.append(valid.sum(dim=1).detach().cpu())

    lattice_feat = torch.cat(lattices, dim=0)
    sg = torch.cat(space_groups).long()
    counts = torch.cat(atom_counts).long().clamp(1, cfg.max_atoms)
    # Une ligne globale garde ensemble groupe d'espace, maille et nombre de sites.
    global_rows = torch.cat(
        [sg.float().unsqueeze(1), lattice_feat.float(), counts.float().unsqueeze(1)],
        dim=1,
    )
    return {
        "representation": "bounded_features_v4_correlated_decode",
        "element_ids": torch.cat(elements).long(),
        "wyckoff_ids": torch.cat(wyckoffs).long(),
        "frac_coords": torch.cat(coords).float(),
        "space_groups": sg,
        "lattice_feat": lattice_feat.float(),
        "atom_counts": counts,
        "global_rows": global_rows.float(),
    }


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
    features = torch.cat([global_feat, sites.flatten(1)], dim=-1)
    return _to_diffusion_range(features)


def diffusion_vector_to_batch(
    vector: torch.Tensor,
    cfg: Config,
    min_atoms: int = 1,
    metadata: dict | None = None,
) -> dict:
    """Discrétise des samples DDPM en batch compatible avec SupraJEPA."""
    shape = diffusion_shape(cfg)
    if vector.shape[-1] != shape.dim:
        raise ValueError(f"Expected diffusion dim {shape.dim}, got {vector.shape[-1]}")

    B = vector.shape[0]
    device = vector.device

    features = _from_diffusion_range(vector)
    global_feat = features[:, : shape.n_global]
    site_feat = features[:, shape.n_global :].reshape(B, shape.n_sites, shape.n_site_features)

    if metadata is not None and "global_rows" in metadata:
        # Score global unique : on trie les prototypes par groupe d'espace,
        # puis on sélectionne une ligne réelle (SG, maille, n_sites).
        global_prob = global_feat.mean(dim=1)
        global_rows = _quantile_lookup_rows(metadata["global_rows"], global_prob, sort_col=0)
        sg_id = global_rows[:, 0].long()
        lattice_feat = global_rows[:, 1:7].to(vector.dtype)
        empirical_atom_counts = global_rows[:, 7].long().clamp(min_atoms, cfg.max_atoms)
    elif metadata is not None and "space_groups" in metadata:
        sg_id = _quantile_lookup(metadata["space_groups"], global_feat[:, 0]).long()
        empirical_atom_counts = None
    else:
        sg_id = torch.round(global_feat[:, 0].clamp(1 / 230, 1.0) * (cfg.n_space_groups - 1)).long()
        sg_id = sg_id.clamp(1, cfg.n_space_groups - 1)
        empirical_atom_counts = None

    if metadata is not None and "global_rows" in metadata:
        pass
    elif metadata is not None and "lattice_feat" in metadata:
        lattice_values = metadata["lattice_feat"].to(device)
        lattice_feat = torch.stack(
            [
                _quantile_lookup(lattice_values[:, j], global_feat[:, j + 1])
                for j in range(6)
            ],
            dim=1,
        ).to(vector.dtype)
    else:
        lattice_feat = global_feat[:, 1:].clone()
        lattice_feat[:, :3] = lattice_feat[:, :3].clamp(0.05, 2.5)
        lattice_feat[:, 3:] = lattice_feat[:, 3:].clamp(0.15, 1.0)

    presence_score = site_feat[:, :, 5]
    if empirical_atom_counts is not None:
        atom_counts = empirical_atom_counts
        padding_atoms = torch.zeros(B, cfg.max_atoms, dtype=torch.bool, device=device)
        for row, n_atoms in enumerate(atom_counts.tolist()):
            topk = torch.topk(presence_score[row], k=max(1, n_atoms)).indices
            padding_atoms[row, topk] = True
    elif metadata is not None and "atom_counts" in metadata:
        count_prob = presence_score.mean(dim=1)
        atom_counts = _quantile_lookup(metadata["atom_counts"], count_prob).long()
        atom_counts = atom_counts.clamp(min_atoms, cfg.max_atoms)
        padding_atoms = torch.zeros(B, cfg.max_atoms, dtype=torch.bool, device=device)
        for row, n_atoms in enumerate(atom_counts.tolist()):
            topk = torch.topk(presence_score[row], k=max(1, n_atoms)).indices
            padding_atoms[row, topk] = True
    else:
        padding_atoms = presence_score > 0.5
        if min_atoms > 0:
            topk = torch.topk(presence_score, k=min(min_atoms, cfg.max_atoms), dim=1).indices
            padding_atoms.scatter_(1, topk, True)

    if metadata is not None and "element_ids" in metadata:
        element_atoms = _quantile_lookup(metadata["element_ids"], site_feat[:, :, 0]).long()
    else:
        element_atoms = torch.round(site_feat[:, :, 0].clamp(1 / 118, 1.0) * (cfg.n_elements - 1)).long()
        element_atoms = element_atoms.clamp(1, cfg.n_elements - 1)

    if metadata is not None and "wyckoff_ids" in metadata:
        wyckoff_atoms = _quantile_lookup(metadata["wyckoff_ids"], site_feat[:, :, 1]).long()
    else:
        wyckoff_atoms = torch.round(site_feat[:, :, 1].clamp(1 / 26, 1.0) * (cfg.n_wyckoff - 1)).long()
        wyckoff_atoms = wyckoff_atoms.clamp(1, cfg.n_wyckoff - 1)
    if metadata is not None and "frac_coords" in metadata:
        coord_values = metadata["frac_coords"].to(device)
        coords_atoms = torch.stack(
            [
                _quantile_lookup(coord_values[:, j], site_feat[:, :, j + 2])
                for j in range(3)
            ],
            dim=-1,
        ).to(vector.dtype)
    else:
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
