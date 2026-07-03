"""Shared small building blocks for property-JEPA (stage 2) modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple SiLU-activated MLP.

    Mirrors the stage-1 block (`crystal_structure_jepa.JEPA.crystal_jepa.MLP`)
    but is reimplemented locally here so that `property_jepa` has no
    import-time dependency on `crystal_structure_jepa` -- the two stages are
    kept independently importable/testable per the repo's stage1/stage2
    module separation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layers: int = 2,
    ):
        super().__init__()
        if layers < 2:
            raise ValueError("MLP requires layers >= 2")
        modules: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(layers - 2):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        modules.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
