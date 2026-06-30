from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class regulizers(nn.Module, ABC):
    """Abstract base class shared by JEPA regularizers.

    The module name follows the existing project spelling, while the class gives
    regularizers the same shared metadata/metric pattern as losses.
    """

    def __init__(self, name=None, weight=1.0, reduction="mean"):
        super().__init__()
        self.name = self.__class__.__name__ if name is None else name
        self.weight = weight
        self.reduction = reduction

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Compute the regularization objective."""

    @staticmethod
    def metrics(**items):
        return {
            key: value.detach().item() if torch.is_tensor(value) else value
            for key, value in items.items()
        }

    def extra_repr(self):
        return f"name={self.name}, weight={self.weight}, reduction={self.reduction}"


class VCLoss(regulizers):
    """Variance-Covariance regularizer attracting covariance to identity."""

    def __init__(self, std_coeff, cov_coeff, proj=None):
        super().__init__(name="VCLoss")
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.proj = nn.Identity() if proj is None else proj
        self.std_loss_fn = HingeStdLoss(std_margin=1.0)
        self.cov_loss_fn = CovarianceLoss()

    def forward(self, x, actions=None):
        del actions
        x = x.transpose(0, 1).flatten(1).transpose(0, 1)  # [B*T*H*W, C]
        fx = self.proj(x)  # [B*T*H*W, C']

        std_loss = self.std_loss_fn(fx)
        cov_loss = self.cov_loss_fn(fx)

        loss = self.std_coeff * std_loss + self.cov_coeff * cov_loss
        total_unweighted_loss = std_loss + cov_loss
        loss_dict = self.metrics(std_loss=std_loss, cov_loss=cov_loss)
        return loss, total_unweighted_loss, loss_dict


class HingeStdLoss(regulizers):
    def __init__(
        self,
        std_margin: float = 1.0,
    ):
        """
        Encourages each feature to maintain at least a minimum standard deviation.
        Features with std below the margin incur a penalty of (std_margin - std).
        Args:
            std_margin (float, default=1.0):
                Minimum desired standard deviation per feature.
        """
        super().__init__(name="HingeStdLoss")
        self.std_margin = std_margin

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        Returns:
            std_loss: Scalar tensor with the hinge loss on standard deviations
        """
        x = x - x.mean(dim=0, keepdim=True)
        std = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(self.std_margin - std))
        return std_loss


class CovarianceLoss(regulizers):
    def __init__(self):
        """
        Penalizes off-diagonal elements of the covariance matrix to encourage
        feature decorrelation.

        Normalizes by D * (D - 1) where D is feature dimensionality.
        """
        super().__init__(name="CovarianceLoss")

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [N, D] where N is number of samples, D is feature dimension
        """
        batch_size = x.shape[0]
        x = x - x.mean(dim=0, keepdim=True)
        cov = (x.T @ x) / (batch_size - 1)  # [D, D]
        return self.off_diagonal(cov).pow(2).mean()


class TemporalSimilarityLoss(regulizers):
    def __init__(self):
        """
        Temporal Similarity Loss.
        Encourages consecutive frames to have similar representations by penalizing
        the squared difference between consecutive time steps.
        """
        super().__init__(name="TemporalSimilarityLoss")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, N, D] where T is time steps, N is batch size, D is feature dimension
        """
        if x.shape[0] <= 1:
            return torch.tensor(0.0, device=x.device)
        sim_loss_t = (x[1:] - x[:-1]).pow(2).mean()
        return sim_loss_t


class InverseDynamicsLoss(regulizers):
    def __init__(self, idm: nn.Module):
        """
        Predicts actions from consecutive states and compares with ground truth actions.
        Args:
            idm (nn.Module): Inverse dynamics model that takes (state_t, state_t+1) and predicts action
        """
        super().__init__(name="InverseDynamicsLoss")
        self.idm = idm

    def forward(self, x: torch.Tensor, actions: torch.Tensor):
        """
        Args:
            x: [T, B, D] - States across time steps
            actions: [B, A, T] - Ground truth actions between consecutive states
        """
        if x.shape[0] <= 1 or actions is None:
            return torch.tensor(0.0, device=x.device)

        _, _, d = x.shape

        states_t = x[:-1].transpose(0, 1)  # [B, T-1, D]
        states_t_plus_1 = x[1:].transpose(0, 1)  # [B, T-1, D]

        states_t_flat = states_t.reshape(-1, d)  # [B*(T-1), D]
        states_t_plus_1_flat = states_t_plus_1.reshape(-1, d)  # [B*(T-1), D]

        pred_actions = self.idm(states_t_flat, states_t_plus_1_flat)  # [B*(T-1), A]
        target_actions = actions.transpose(1, 2)[:, :-1].reshape(
            -1, actions.size(1)
        )  # [B*(T-1), A]
        idm_loss = F.mse_loss(pred_actions, target_actions)

        return idm_loss


class VC_IDM_Sim_Regularizer(regulizers):
    def __init__(
        self,
        cov_coeff: float,
        std_coeff: float,
        sim_coeff_t: float,
        idm_coeff: float = 0.0,
        idm: nn.Module = None,
        std_margin: float = 1,
        first_t_only: bool = True,
        projector: nn.Module = None,
        spatial_as_samples: bool = False,
        sim_t_after_proj: bool = False,
        idm_after_proj: bool = False,
    ):
        """
        Composite regularizer combining:
        - Hinge Standard Deviation Loss
        - Covariance Decorrelation Loss
        - Temporal Similarity Loss
        - Inverse Dynamics Model Loss
        """
        super().__init__(name="VC_IDM_Sim_Regularizer")
        self.cov_coeff = cov_coeff
        self.std_coeff = std_coeff
        self.sim_coeff_t = sim_coeff_t
        self.idm_coeff = idm_coeff

        self.first_t_only = first_t_only
        self.projector = nn.Identity() if projector is None else projector
        self.spatial_as_samples = spatial_as_samples
        self.sim_t_after_proj = sim_t_after_proj
        self.idm_after_proj = idm_after_proj

        self.std_loss_fn = HingeStdLoss(std_margin=std_margin)
        self.cov_loss_fn = CovarianceLoss()
        self.sim_loss_fn = TemporalSimilarityLoss()
        self.idm_loss_fn = InverseDynamicsLoss(idm) if idm is not None else None

    def forward(self, x, actions=None):
        """
        Args:
            x: [B, C, T, H, W] - Input activations. Internally reshaped to either
                [1, B, D] when first_t_only=True or [T*B, D] otherwise, with D=C*H*W.
            actions: [B, A, T] - Optional actions for IDM loss
        """
        b, c, t, h, w = x.shape

        x_unprojected = x.permute(2, 0, 1, 3, 4).reshape(t, b, -1)  # [T, B, C*H*W]

        x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, c)  # [B*T*H*W, C]
        x_proj = self.projector(x_flat)  # [B*T*H*W, C_out]
        c_out = x_proj.shape[-1]
        x_projected = x_proj.view(b, t, h, w, c_out)  # [B, T, H, W, C_out]
        x_projected_reshaped = x_projected.permute(2, 0, 1, 3, 4).reshape(
            t, b, -1
        )  # [T, B, C_out*H*W]

        if self.sim_t_after_proj:
            sim_loss_t = self.sim_loss_fn(x_projected_reshaped)
        else:
            sim_loss_t = self.sim_loss_fn(x_unprojected)

        idm_loss = torch.tensor(0.0, device=x.device)
        if self.idm_coeff > 0 and self.idm_loss_fn is not None and actions is not None:
            if self.idm_after_proj:
                idm_loss = self.idm_loss_fn(x_projected_reshaped, actions)
            else:
                idm_loss = self.idm_loss_fn(x_unprojected, actions)

        if self.spatial_as_samples:
            if self.first_t_only:
                x_for_vc = x_projected[:, 0].reshape(b * h * w, c_out)
                assert x_for_vc.shape == (b * h * w, c_out)
            else:
                x_for_vc = x_projected.reshape(-1, c_out)
                assert x_for_vc.shape == (b * t * h * w, c_out)
        else:
            x_for_vc = x_projected.permute(0, 1, 4, 2, 3).reshape(
                b, t, -1
            )  # [B, T, C_out*H*W]
            if self.first_t_only:
                x_for_vc = x_for_vc[:, 0]
                assert x_for_vc.shape == (b, c_out * h * w)
            else:
                x_for_vc = x_for_vc.reshape(-1, x_for_vc.size(-1))
                assert x_for_vc.shape == (b * t, c_out * h * w)

        std_loss = self.std_loss_fn(x_for_vc)
        cov_loss = self.cov_loss_fn(x_for_vc)

        total_weighted_loss = (
            self.cov_coeff * cov_loss
            + self.std_coeff * std_loss
            + self.sim_coeff_t * sim_loss_t
            + self.idm_coeff * idm_loss
        )
        total_unweighted_loss = cov_loss + std_loss + sim_loss_t + idm_loss

        loss_dict = self.metrics(
            cov_loss=cov_loss,
            std_loss=std_loss,
            sim_loss_t=sim_loss_t,
            idm_loss=idm_loss,
        )

        return total_weighted_loss, total_unweighted_loss, loss_dict


class VISRegLoss(regulizers):
    """Variance-Invariance-Sketching regularization for JEPA training.

    Implements the VISReg objective from arXiv:2606.02572:
    variance/scale regularization, Sliced-Wasserstein shape regularization, and
    an optional centering term. When two views are provided, an invariance MSE
    term is added to form the full VISReg loss.
    """

    def __init__(
        self,
        num_slices=64,
        scale_coeff=1.0,
        shape_coeff=1.0,
        center_coeff=1.0,
        invariance_coeff=1.0,
        eps=1e-4,
        deterministic_slices=False,
        name=None,
        reduction="mean",
    ):
        super().__init__(
            name="VISRegLoss" if name is None else name,
            weight=1.0,
            reduction=reduction,
        )
        self.num_slices = num_slices
        self.scale_coeff = scale_coeff
        self.shape_coeff = shape_coeff
        self.center_coeff = center_coeff
        self.invariance_coeff = invariance_coeff
        self.eps = eps
        self.deterministic_slices = deterministic_slices
        self.step = 0

    def _flatten_embeddings(self, z):
        """Return embeddings as [N, D].

        For JEPA feature maps [B, C, T, H, W], spatial and temporal locations are
        treated as samples and channels as feature dimensions.
        """
        if z.dim() == 2:
            return z
        if z.dim() == 5:
            b, c, t, h, w = z.shape
            return z.permute(0, 2, 3, 4, 1).reshape(b * t * h * w, c)
        return z.reshape(-1, z.shape[-1])

    def _random_directions(self, num_features, device, dtype):
        generator = None
        if self.deterministic_slices:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.step)
            self.step += 1

        directions = torch.randn(
            num_features,
            self.num_slices,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return F.normalize(directions, p=2, dim=0, eps=self.eps)

    def _normal_quantiles(self, num_samples, device, dtype):
        u = torch.arange(1, num_samples + 1, device=device, dtype=dtype)
        u = u / (num_samples + 1)
        normal = Normal(
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
        )
        return normal.icdf(u).view(num_samples, 1)

    def regularization(self, z):
        """Compute VISReg scale, shape, and center terms for one embedding set."""
        z = self._flatten_embeddings(z)
        if z.size(0) < 2:
            zero = z.new_tensor(0.0)
            return zero, zero, zero, zero

        mean = z.mean(dim=0, keepdim=True)
        center_loss = mean.pow(2).mean()

        z_centered = z - mean
        std = z_centered.std(dim=0, unbiased=False)
        scale_loss = (1.0 - std).pow(2).mean()

        z_normalized = z_centered / (std.detach().clamp_min(self.eps))
        directions = self._random_directions(z.size(1), z.device, z.dtype)
        projected = z_normalized @ directions
        projected = projected.sort(dim=0).values
        target = self._normal_quantiles(z.size(0), z.device, z.dtype)
        shape_loss = (projected - target).pow(2).mean()

        reg_loss = (
            self.scale_coeff * scale_loss
            + self.shape_coeff * shape_loss
            + self.center_coeff * center_loss
        )
        return reg_loss, scale_loss, shape_loss, center_loss

    def forward(self, z1, z2=None):
        """Compute VISReg loss.

        Args:
            z1: Embeddings or JEPA feature maps.
            z2: Optional second view/prediction. If provided, the returned loss
                includes MSE invariance plus regularization on both views.
        """
        reg1, scale1, shape1, center1 = self.regularization(z1)

        invariance_loss = z1.new_tensor(0.0)
        if z2 is None:
            reg_loss = reg1
            scale_loss = scale1
            shape_loss = shape1
            center_loss = center1
        else:
            z1_flat = self._flatten_embeddings(z1)
            z2_flat = self._flatten_embeddings(z2)
            if z1_flat.shape != z2_flat.shape:
                raise ValueError(
                    f"VISRegLoss expected matching flattened shapes, got "
                    f"{tuple(z1_flat.shape)} and {tuple(z2_flat.shape)}"
                )
            invariance_loss = F.mse_loss(z1_flat, z2_flat, reduction=self.reduction)
            reg2, scale2, shape2, center2 = self.regularization(z2)
            reg_loss = 0.5 * (reg1 + reg2)
            scale_loss = 0.5 * (scale1 + scale2)
            shape_loss = 0.5 * (shape1 + shape2)
            center_loss = 0.5 * (center1 + center2)

        total_loss = self.invariance_coeff * invariance_loss + reg_loss

        return {
            "loss": total_loss,
            "invariance_loss": invariance_loss,
            "reg_loss": reg_loss,
            "scale_loss": scale_loss,
            "shape_loss": shape_loss,
            "center_loss": center_loss,
        }


class VISRegRegularizer(VISRegLoss):
    """JEPA regularizer wrapper returning the tuple expected by ``JEPA.unroll``."""

    def forward(self, x, actions=None):
        del actions
        out = super().forward(x)
        unweighted = out["scale_loss"] + out["shape_loss"] + out["center_loss"]
        return (
            out["loss"],
            unweighted,
            self.metrics(
                scale_loss=out["scale_loss"],
                shape_loss=out["shape_loss"],
                center_loss=out["center_loss"],
                reg_loss=out["reg_loss"],
            ),
        )
