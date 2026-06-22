"""Baseline deterministic Tc heuristic."""

from __future__ import annotations

from crys_jepa_gnome_sc.chemistry import METALS, parse_formula
from crys_jepa_gnome_sc.models import Candidate, TcPrediction

RARE_EARTHS = {"Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Dy", "Lu"}
ALKALINE_EARTHS = {"Mg", "Ca", "Sr", "Ba"}
TRANSITION_METALS = {"Fe", "Co", "Ni", "Cu", "Nb", "Mo", "Pd", "Ta", "W"}
PNICTOGENS_CHALCOGENS = {"N", "P", "As", "S", "Se", "Te"}


def _stability_bonus(candidate: Candidate) -> float:
    if candidate.energy_above_hull is None:
        return -2.0
    return max(-10.0, 8.0 - candidate.energy_above_hull * 45.0)


def predict_tc(candidate: Candidate) -> TcPrediction:
    """Predict a baseline superconducting transition temperature in Kelvin."""

    composition = parse_formula(candidate.formula)
    if not composition.valid:
        return TcPrediction(kelvin=0.0, confidence=0.0, factors=["invalid_formula"])

    elements = set(composition.elements)
    tc = 1.0
    factors = ["base_prior"]

    h_fraction = composition.fraction("H")
    if "H" in elements and h_fraction >= 0.55 and elements & RARE_EARTHS:
        tc += 155.0 * h_fraction
        factors.append("rare_earth_hydride")
    elif "H" in elements:
        tc += 35.0 * h_fraction
        factors.append("hydrogen_rich")

    if "B" in elements and elements & ALKALINE_EARTHS:
        tc += 38.0
        factors.append("alkaline_earth_boride")

    if {"Cu", "O"} <= elements and elements & {"Y", "La", "Ba", "Sr", "Ca"}:
        tc += 82.0
        factors.append("cuprate_like")

    if "Fe" in elements and elements & PNICTOGENS_CHALCOGENS:
        tc += 32.0
        factors.append("iron_pnictide_chalcogenide_like")

    if elements & {"C", "N"} and elements & TRANSITION_METALS:
        tc += 12.0
        factors.append("transition_metal_light_element")

    if candidate.is_metallic is True or (candidate.band_gap is not None and candidate.band_gap <= 0.25):
        tc += 8.0
        factors.append("metallic_or_low_gap")
    elif candidate.band_gap is not None:
        tc -= min(25.0, candidate.band_gap * 5.0)
        factors.append("wide_gap_penalty")

    tc += _stability_bonus(candidate)
    if elements & METALS:
        tc += 3.0
        factors.append("contains_metal")

    kelvin = round(max(0.0, tc), 3)
    confidence = 0.35
    if candidate.energy_above_hull is not None:
        confidence += 0.20
    if candidate.band_gap is not None or candidate.is_metallic is not None:
        confidence += 0.20
    if len(elements) >= 2:
        confidence += 0.10
    return TcPrediction(kelvin=kelvin, confidence=min(1.0, confidence), factors=factors)
