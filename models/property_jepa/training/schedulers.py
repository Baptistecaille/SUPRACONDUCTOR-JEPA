"""Learning-rate scheduler for property-JEPA (stage 2) training.

Reimplemented locally from stage 1's `CosineWithWarmup`
(`models/crystal_structure_jepa/JEPA/schedulers.py`) to keep the two stages
independently importable (see the module docstrings in
`models/property_jepa/model/losses.py` and `regulizers.py` for the same
rationale).
"""

from __future__ import annotations

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class CosineWithWarmup:
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self, optimizer, total_steps, warmup_ratio=0.1, min_lr=1e-5, last_epoch=-1
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if not 0 <= warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")

        self.warmup_steps = int(warmup_ratio * total_steps)
        self.cosine_steps = total_steps - self.warmup_steps

        if self.warmup_steps == 0:
            self.scheduler = CosineAnnealingLR(
                optimizer, T_max=self.cosine_steps, eta_min=min_lr
            )
            return

        warmup = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=self.warmup_steps
        )
        cosine = CosineAnnealingLR(optimizer, T_max=self.cosine_steps, eta_min=min_lr)

        self.scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_steps]
        )

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)
