"""Transformer backbone for `foundation_jepa`, reimplementing the official
Crys-JEPA architecture (arXiv:2605.14759, `github.com/liun-online/Crys_JEPA`,
`components/jepa/backbone/transformer.py` + `components/jepa/frame/jepa.py`)
without a `torch_geometric` dependency.

The official code batches variable-atom-count crystals via
`torch_geometric.utils.to_dense_batch` (a `[N_total, F]` ragged tensor + a
`batch` index vector -> dense `[B, N_max, F]` + boolean mask). This repo's
sibling modules (`models/property_jepa/model/crystal_encoder.py`,
`models/crystal_structure_jepa/JEPA/crystal_jepa.py`) instead pad in the
collate function and pass an already-dense `[B, N_max, F]` tensor + `[B,
N_max]` mask straight into the model -- functionally identical (both produce
a dense batch + mask), but avoids adding a heavy graph-learning dependency
for what is, here, just a padding operation. `PreNormEncoderLayer` /
`FoundationTransformer` below are a faithful port of the official
`Decoder_layer` / `Transformer` classes (same pre-LN self-attention + MLP
block structure, same learned absolute positional embedding table, same CLS
token prepended before the stack, same "zero out padding after every block"
masking discipline) operating on that pre-padded input instead.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MLP(nn.Module):
    """SiLU-activated MLP, port of the official `components.jepa.backbone.transformer.MLP`."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
        super().__init__()
        if n_layers < 2:
            raise ValueError("MLP requires n_layers >= 2")
        self.map = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(n_layers - 2):
            self.map.append(nn.Linear(hidden_dim, hidden_dim))
        self.map.append(nn.Linear(hidden_dim, out_dim))
        self.act = nn.SiLU()
        self.n_layers = n_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.n_layers - 1):
            x = self.act(self.map[i](x))
        return self.map[-1](x)


class MaskedMHA(nn.Module):
    """Multi-head self-attention with a `[B, L]` padding mask, port of official `MHA`."""

    def __init__(self, attn_head: int, dim: int, dropout: float):
        super().__init__()
        if dim % attn_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by attn_head ({attn_head})")
        self.dropout = nn.Dropout(dropout)
        self.dim = dim
        self.attn_head = attn_head
        self.softmax = nn.Softmax(dim=-1)
        self.WO = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, lq, _ = q.shape
        _, lk, _ = k.shape
        _, lv, _ = v.shape
        dim_head = self.dim // self.attn_head

        q = q.reshape(b, lq, self.attn_head, dim_head).transpose(1, 2)
        k = k.reshape(b, lk, self.attn_head, dim_head).transpose(1, 2)
        v = v.reshape(b, lv, self.attn_head, dim_head).transpose(1, 2)

        pair_mask = mask.unsqueeze(1).unsqueeze(-1).float()
        pair_mask = pair_mask @ pair_mask.transpose(-1, -2)

        attn_scores = q @ k.transpose(2, 3) / math.sqrt(dim_head)
        attn_scores = attn_scores.masked_fill(pair_mask == 0, float("-1e3"))
        attn = self.softmax(attn_scores)
        attn = self.dropout(attn)

        attn_out = (attn @ v).transpose(1, 2).reshape(b, lq, self.dim)
        attn_out = self.dropout(attn_out)
        return self.WO(attn_out)


class PreNormEncoderLayer(nn.Module):
    """Pre-LN self-attention + MLP block, port of official `Decoder_layer`."""

    def __init__(self, dim: int, attn_head: int, dropout: float):
        super().__init__()
        if dim % attn_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by attn_head ({attn_head})")
        self.ln_attn = nn.LayerNorm(dim)
        self.ln_mlp = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.mha = MaskedMHA(attn_head, dim, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        h_in = h
        h_norm = self.ln_attn(h)
        q, k, v = self.qkv(h_norm).chunk(3, dim=-1)
        attn_out = self.mha(q, k, v, mask)
        attn_out = attn_out.masked_fill(~mask.unsqueeze(-1), 0.0)
        h = h_in + attn_out
        h = h + self.mlp(self.ln_mlp(h))
        h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        return h


class FoundationTransformer(nn.Module):
    """CLS-token transformer stack, port of official `Transformer`.

    Args:
        hidden_dim: Embedding width.
        layers: Number of `PreNormEncoderLayer` blocks.
        attn_head: Number of attention heads (must divide `hidden_dim`).
        dropout: Dropout probability inside attention/MLP.
        max_len: Max sequence length (atoms + CLS token) covered by the
            learned absolute positional embedding table.
    """

    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        attn_head: int,
        dropout: float,
        max_len: int = 500,
    ):
        super().__init__()
        self.pe_emb = nn.Parameter(torch.zeros(max_len, hidden_dim))
        nn.init.trunc_normal_(self.pe_emb, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList(
            [PreNormEncoderLayer(hidden_dim, attn_head, dropout) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Per-atom token embeddings, `[B, N, hidden_dim]` (CLS NOT included).
            mask: Boolean atom mask, `[B, N]` (CLS NOT included; True = real atom).

        Returns:
            `[B, N+1, hidden_dim]`, position 0 is the CLS token's final embedding.
        """
        b, n = mask.shape
        if n + 1 > self.pe_emb.shape[0]:
            raise ValueError(
                f"sequence length {n + 1} exceeds positional embedding table "
                f"size {self.pe_emb.shape[0]}"
            )
        cls_tokens = self.cls_token.expand(b, 1, -1)
        h = torch.cat([cls_tokens, h], dim=1)
        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)

        h = h + self.pe_emb[: n + 1].unsqueeze(0)

        for block in self.blocks:
            h = h.masked_fill(~full_mask.unsqueeze(-1), 0.0)
            h = block(h, full_mask)
        h = self.norm(h)
        h = h.masked_fill(~full_mask.unsqueeze(-1), 0.0)
        return h


__all__ = ["MLP", "MaskedMHA", "PreNormEncoderLayer", "FoundationTransformer"]
