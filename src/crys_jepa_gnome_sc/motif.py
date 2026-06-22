"""DINO-crystal-inspired motif scoring."""

from __future__ import annotations

from crys_jepa_gnome_sc.chemistry import parse_formula
from crys_jepa_gnome_sc.models import Candidate, MotifScore


def score_motifs(candidate: Candidate) -> MotifScore:
    """Score superconductivity-associated motif families."""

    composition = parse_formula(candidate.formula)
    if not composition.valid:
        return MotifScore(0.0, ["invalid_formula"])

    elements = set(composition.elements)
    score = 0.05
    motifs = []

    if "H" in elements and composition.fraction("H") >= 0.55:
        score += 0.38
        motifs.append("hydride_rich")
    if "B" in elements and elements & {"Mg", "Ca", "Sr", "Ba", "Y", "La"}:
        score += 0.25
        motifs.append("boride_network")
    if {"Cu", "O"} <= elements and elements & {"Y", "La", "Ba", "Sr", "Ca"}:
        score += 0.34
        motifs.append("cuprate_like")
    if "Fe" in elements and elements & {"As", "P", "Se", "Te", "S"}:
        score += 0.28
        motifs.append("iron_pnictide_chalcogenide")
    if elements & {"C", "N"} and elements & {"Nb", "Ti", "V", "Mo", "W"}:
        score += 0.16
        motifs.append("carbide_nitride")
    if "O" in elements and len(elements) >= 4:
        score += 0.08
        motifs.append("complex_layered_oxide")

    if not motifs:
        motifs.append("no_known_superconducting_motif")
    return MotifScore(score=round(min(1.0, score), 3), motifs=motifs)
