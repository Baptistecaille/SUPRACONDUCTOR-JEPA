"""Shared data models for the screening pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Candidate:
    """A GNoME-like crystal candidate row."""

    material_id: str
    formula: str
    formation_energy_per_atom: Optional[float] = None
    energy_above_hull: Optional[float] = None
    band_gap: Optional[float] = None
    density: Optional[float] = None
    space_group: Optional[int] = None
    volume: Optional[float] = None
    num_sites: Optional[int] = None
    is_metallic: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FormulaComposition:
    """Parsed formula composition."""

    formula: str
    amounts: Dict[str, float]
    valid: bool
    error: Optional[str] = None

    @property
    def total_atoms(self) -> float:
        return sum(self.amounts.values())

    @property
    def elements(self) -> List[str]:
        return list(self.amounts)

    def fraction(self, element: str) -> float:
        if self.total_atoms <= 0:
            return 0.0
        return self.amounts.get(element, 0.0) / self.total_atoms


@dataclass(frozen=True)
class FilterResult:
    """Chemical and stability filter result."""

    passed: bool
    score: float
    reasons: List[str]
    composition: FormulaComposition


@dataclass(frozen=True)
class TcPrediction:
    """Baseline transition-temperature prediction."""

    kelvin: float
    confidence: float
    factors: List[str]


@dataclass(frozen=True)
class GraphSummary:
    """Compact crystal graph feature summary."""

    nodes: int
    edges: int
    average_coordination: float
    periodicity_confidence: float
    features: Dict[str, float]


@dataclass(frozen=True)
class SurpriseScore:
    """Crys-JEPA-inspired novelty score."""

    score: float
    reasons: List[str]


@dataclass(frozen=True)
class MotifScore:
    """DINO-crystal-inspired motif score."""

    score: float
    motifs: List[str]


@dataclass(frozen=True)
class RankedCandidate:
    """Final ranked output row."""

    rank: int
    material_id: str
    formula: str
    passed_filter: bool
    rank_score: float
    predicted_tc_k: float
    filter_score: float
    surprise_score: float
    motif_score: float
    graph_confidence: float
    synthesizability_score: float
    reasons: List[str]
