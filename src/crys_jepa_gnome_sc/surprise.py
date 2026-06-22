"""Crys-JEPA-inspired latent surprise heuristic."""

from __future__ import annotations

from crys_jepa_gnome_sc.chemistry import parse_formula
from crys_jepa_gnome_sc.models import Candidate, FilterResult, SurpriseScore


def score_surprise(candidate: Candidate, filter_result: FilterResult) -> SurpriseScore:
    """Score unusual but plausible candidate chemistry and structure."""

    composition = parse_formula(candidate.formula)
    if not composition.valid:
        return SurpriseScore(0.0, ["invalid_formula"])

    elements = set(composition.elements)
    score = 0.20
    reasons = ["base_novelty_prior"]

    if filter_result.passed:
        score += 0.15
        reasons.append("plausible_after_filter")
    else:
        score -= 0.15
        reasons.append("filter_penalty")

    if len(elements) >= 4:
        score += 0.15
        reasons.append("multi_element_complexity")
    if "H" in elements and composition.fraction("H") >= 0.5:
        score += 0.18
        reasons.append("hydrogen_rich_latent_region")
    if {"Cu", "O"} <= elements:
        score += 0.12
        reasons.append("oxide_correlation")
    if {"Fe"} & elements and elements & {"As", "Se", "P", "Te"}:
        score += 0.14
        reasons.append("iron_layer_candidate")
    if candidate.space_group is not None and candidate.space_group > 0:
        score += 0.07
        reasons.append("has_symmetry_context")
    if candidate.energy_above_hull is not None:
        score += max(0.0, 0.08 - candidate.energy_above_hull * 0.25)

    return SurpriseScore(score=round(max(0.0, min(1.0, score)), 3), reasons=reasons)
