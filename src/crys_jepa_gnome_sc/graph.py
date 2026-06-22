"""Lightweight periodic crystal graph summaries."""

from __future__ import annotations

from crys_jepa_gnome_sc.chemistry import parse_formula
from crys_jepa_gnome_sc.models import Candidate, GraphSummary


def build_graph_summary(candidate: Candidate) -> GraphSummary:
    """Build a compact graph summary from structure fields or formula fallback."""

    composition = parse_formula(candidate.formula)
    formula_atoms = int(round(composition.total_atoms)) if composition.valid else 1
    nodes = candidate.num_sites or max(1, formula_atoms)

    has_structure = sum(
        value is not None
        for value in (candidate.num_sites, candidate.space_group, candidate.volume, candidate.density)
    )
    periodicity_confidence = min(1.0, 0.25 + has_structure * 0.18)

    density_term = 0.0 if candidate.density is None else min(3.0, candidate.density / 3.0)
    space_group_term = 0.0 if candidate.space_group is None else min(2.0, candidate.space_group / 115.0)
    average_coordination = min(12.0, 2.0 + len(composition.elements) * 1.3 + density_term + space_group_term)
    edges = max(nodes - 1, int(round(nodes * average_coordination / 2.0)))

    features = {
        "element_count": float(len(composition.elements)),
        "formula_atoms": float(formula_atoms),
        "density": float(candidate.density or 0.0),
        "volume_per_site": float((candidate.volume or 0.0) / nodes) if nodes else 0.0,
        "space_group": float(candidate.space_group or 0),
    }
    return GraphSummary(
        nodes=nodes,
        edges=edges,
        average_coordination=round(average_coordination, 3),
        periodicity_confidence=round(periodicity_confidence, 3),
        features=features,
    )
