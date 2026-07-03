"""Public API for the property-JEPA (stage 2) model architecture."""

from .crystal_encoder import (
    LATTICE_FEATURE_DIM,
    NUM_ATOM_CLASSES,
    RotationInvariantCrystalEncoder,
    build_atom_tokens,
)
from .jepa import PropertyJEPA, PropertyJEPAOutput
from .layers import MLP
from .losses import PropertyJEPALoss, property_jepa_loss
from .predictor import PropertyPredictor
from .property_encoder import PropertyEncoder, TargetPropertyEncoder, masked_mean_pool
from .regulizers import CovarianceLoss, HingeStdLoss, VCLoss

__all__ = [
    "LATTICE_FEATURE_DIM",
    "NUM_ATOM_CLASSES",
    "RotationInvariantCrystalEncoder",
    "build_atom_tokens",
    "PropertyJEPA",
    "PropertyJEPAOutput",
    "MLP",
    "PropertyJEPALoss",
    "property_jepa_loss",
    "PropertyPredictor",
    "PropertyEncoder",
    "TargetPropertyEncoder",
    "masked_mean_pool",
    "CovarianceLoss",
    "HingeStdLoss",
    "VCLoss",
]
