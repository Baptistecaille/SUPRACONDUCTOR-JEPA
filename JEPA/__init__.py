"""Public API for the EB-JEPA library."""

import sys
from importlib.metadata import PackageNotFoundError, version

from .architectures import (
    DetHead,
    ImpalaEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    ResNet5,
    ResUNet,
    ResidualBlock,
    ResnetBlock,
    ResnetStack,
    SimplePredictor,
    StateOnlyPredictor,
    conv3d2,
)
from .image_decoder import ImageDecoder
from .crystal_jepa import (
    CrystalJEPA,
    CrystalTransformerEncoder,
    MaskConditionedPredictor,
    TargetCrystalTransformerEncoder,
)
from .jepa import JEPA, JEPAbase, JEPAProbe
from .losses import (
    BCS,
    SquareLossSeq,
    VICRegLoss,
    WeightedContrastiveLoss,
    losses,
    sq_loss,
    square_cost_seq,
    weighted_contrastive_loss,
)
from .regulizers import (
    CovarianceLoss,
    HingeStdLoss,
    InverseDynamicsLoss,
    TemporalSimilarityLoss,
    VC_IDM_Sim_Regularizer,
    VCLoss,
    VISRegLoss,
    VISRegRegularizer,
    regulizers,
)
from .nn_utils import TemporalBatchMixin, init_module_weights
from .schedulers import CosineWithWarmup
from .state_decoder import MLPXYHead

try:
    __version__ = version("supra-jepa")
except PackageNotFoundError:
    __version__ = "0.0.0+editable"

sys.modules.setdefault("eb_jepa", sys.modules[__name__])

__all__ = [
    "__version__",
    "BCS",
    "CosineWithWarmup",
    "CovarianceLoss",
    "CrystalJEPA",
    "CrystalTransformerEncoder",
    "DetHead",
    "HingeStdLoss",
    "ImageDecoder",
    "ImpalaEncoder",
    "InverseDynamicsLoss",
    "InverseDynamicsModel",
    "JEPA",
    "JEPAbase",
    "JEPAProbe",
    "MaskConditionedPredictor",
    "MLPXYHead",
    "Projector",
    "RNNPredictor",
    "ResNet5",
    "ResUNet",
    "ResidualBlock",
    "ResnetBlock",
    "ResnetStack",
    "SimplePredictor",
    "SquareLossSeq",
    "StateOnlyPredictor",
    "TargetCrystalTransformerEncoder",
    "TemporalBatchMixin",
    "TemporalSimilarityLoss",
    "VC_IDM_Sim_Regularizer",
    "VCLoss",
    "VICRegLoss",
    "VISRegLoss",
    "VISRegRegularizer",
    "WeightedContrastiveLoss",
    "conv3d2",
    "init_module_weights",
    "losses",
    "regulizers",
    "sq_loss",
    "square_cost_seq",
    "weighted_contrastive_loss",
]
