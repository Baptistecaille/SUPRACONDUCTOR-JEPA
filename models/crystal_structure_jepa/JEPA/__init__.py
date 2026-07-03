"""Crystal JEPA components for superconducting-material representation learning."""

from importlib.metadata import PackageNotFoundError, version

from .crystal_jepa import (
    CrystalJEPA,
    CrystalTransformerEncoder,
    MaskConditionedPredictor,
    TargetCrystalTransformerEncoder,
)
from .losses import WeightedContrastiveLoss, losses, weighted_contrastive_loss
from .regulizers import CovarianceLoss, HingeStdLoss, VCLoss, regulizers
from .schedulers import CosineWithWarmup

try:
    __version__ = version("supra-jepa")
except PackageNotFoundError:
    __version__ = "0.0.0+editable"

__all__ = [
    "__version__",
    "CosineWithWarmup",
    "CovarianceLoss",
    "CrystalJEPA",
    "CrystalTransformerEncoder",
    "HingeStdLoss",
    "MaskConditionedPredictor",
    "TargetCrystalTransformerEncoder",
    "VCLoss",
    "WeightedContrastiveLoss",
    "losses",
    "regulizers",
    "weighted_contrastive_loss",
]
