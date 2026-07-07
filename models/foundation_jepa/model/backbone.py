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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint


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
    """Multi-head self-attention with a `[B, L]` padding mask, port of official `MHA`.

    Uses `torch.nn.functional.scaled_dot_product_attention` (SDPA) instead of
    manually materializing the `[B, attn_head, L, L]` score/softmax tensors.
    The manual (matmul -> masked_fill -> softmax -> matmul) formulation used
    prior to this change allocates 3 full `[B, attn_head, L, L]` fp32 tensors
    per layer that must be kept around for backward; at the paper's config
    (`hidden_dim=512` -> `attn_head=16`, `batch_size=2048`, `L~201` atoms+CLS)
    that is ~5.3 GB *per tensor* and blew a 14.56 GiB Colab GPU with a CUDA
    OOM (see `docs/audit/foundation_jepa_pretrain_report.json` run log /
    incident). SDPA dispatches to a fused kernel (Flash-Attention /
    memory-efficient attention on CUDA) that never materializes the full
    `L x L` matrix, so memory scales with `L` instead of `L^2` and this same
    config fits comfortably on a single GPU. Only the key/padding mask is
    passed (not a query-side mask): the CLS token is always unmasked, so no
    key row is ever fully masked out (avoiding the all -inf -> NaN softmax
    edge case), and padded query rows are zeroed by the caller's
    `masked_fill` regardless of what value they compute here.
    """

    def __init__(self, attn_head: int, dim: int, dropout: float):
        super().__init__()
        if dim % attn_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by attn_head ({attn_head})")
        self.dropout_p = dropout
        self.dim = dim
        self.attn_head = attn_head
        self.WO = nn.Linear(dim, dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, lq, _ = q.shape
        _, lk, _ = k.shape
        _, lv, _ = v.shape
        dim_head = self.dim // self.attn_head

        q = q.reshape(b, lq, self.attn_head, dim_head).transpose(1, 2)
        k = k.reshape(b, lk, self.attn_head, dim_head).transpose(1, 2)
        v = v.reshape(b, lv, self.attn_head, dim_head).transpose(1, 2)

        # Key-side padding mask, broadcast over heads and query positions:
        # [B, L] -> [B, 1, 1, L]. True = attend (SDPA boolean-mask convention).
        attn_mask = mask.unsqueeze(1).unsqueeze(1)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout_p if self.training else 0.0
        )
        attn_out = attn_out.transpose(1, 2).reshape(b, lq, self.dim)
        attn_out = F.dropout(attn_out, p=self.dropout_p, training=self.training)
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
        grad_checkpointing: If True, wrap each `PreNormEncoderLayer` block in
            `torch.utils.checkpoint.checkpoint` during training. Trades one
            extra forward pass per block (recompute in backward) for not
            keeping that block's activations resident across the whole
            stack at once -- on an 8-layer stack this turns "8 blocks of
            activations alive simultaneously" into "~1 block alive at a
            time", roughly an 8x cut to the transformer's activation memory
            at the cost of ~20-30% more compute. Only engages in training
            mode with grad enabled; a no-op under `model.eval()` /
            `torch.no_grad()` (no backward pass to save memory for there).
    """

    def __init__(
        self,
        hidden_dim: int,
        layers: int,
        attn_head: int,
        dropout: float,
        max_len: int = 500,
        grad_checkpointing: bool = False,
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
        self.grad_checkpointing = grad_checkpointing

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

        use_checkpoint = self.grad_checkpointing and self.training and torch.is_grad_enabled()
        for block in self.blocks:
            h = h.masked_fill(~full_mask.unsqueeze(-1), 0.0)
            if use_checkpoint:
                h = torch_checkpoint.checkpoint(block, h, full_mask, use_reentrant=False)
            else:
                h = block(h, full_mask)
        h = self.norm(h)
        h = h.masked_fill(~full_mask.unsqueeze(-1), 0.0)
        return h


__all__ = ["MLP", "MaskedMHA", "PreNormEncoderLayer", "FoundationTransformer"]
