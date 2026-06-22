"""Formula parsing and first-pass chemistry filters."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Set

from crys_jepa_gnome_sc.models import Candidate, FilterResult, FormulaComposition

ELEMENTS: Set[str] = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu",
}

RADIOACTIVE_OR_UNSUPPORTED: Set[str] = {
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu",
}

METALS: Set[str] = {
    "Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr",
    "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Rb", "Sr", "Y", "Zr", "Nb",
    "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi",
}

NONMETALS: Set[str] = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Se", "I"}

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)")


def parse_formula(formula: str) -> FormulaComposition:
    """Parse a simple chemical formula such as MgB2 or YBa2Cu3O7."""

    text = (formula or "").strip()
    if not text:
        return FormulaComposition(formula=formula, amounts={}, valid=False, error="empty_formula")
    if any(char in text for char in "()[]{}"):
        return FormulaComposition(formula=formula, amounts={}, valid=False, error="grouped_formula_not_supported")

    amounts: Dict[str, float] = {}
    position = 0
    for match in _TOKEN_RE.finditer(text):
        if match.start() != position:
            return FormulaComposition(formula=formula, amounts={}, valid=False, error="invalid_syntax")
        element, amount_text = match.groups()
        if element not in ELEMENTS:
            return FormulaComposition(formula=formula, amounts={}, valid=False, error="unknown_element")
        amount = float(amount_text) if amount_text else 1.0
        if amount <= 0:
            return FormulaComposition(formula=formula, amounts={}, valid=False, error="invalid_amount")
        amounts[element] = amounts.get(element, 0.0) + amount
        position = match.end()

    if position != len(text) or not amounts:
        return FormulaComposition(formula=formula, amounts={}, valid=False, error="invalid_syntax")
    return FormulaComposition(formula=formula, amounts=amounts, valid=True)


def has_any(elements: Iterable[str], family: Set[str]) -> bool:
    return bool(set(elements) & family)


def apply_chemical_filter(candidate: Candidate, hull_threshold: float = 0.20) -> FilterResult:
    """Apply plausibility, safety, stability, and metallicity filters."""

    composition = parse_formula(candidate.formula)
    reasons = []
    score = 1.0
    passed = True

    if not composition.valid:
        return FilterResult(False, 0.0, [composition.error or "invalid_formula"], composition)

    elements = set(composition.elements)
    if elements & RADIOACTIVE_OR_UNSUPPORTED:
        reasons.append("radioactive_or_unsupported_element")
        score -= 0.75
        passed = False

    if candidate.energy_above_hull is None:
        reasons.append("missing_hull_energy")
        score -= 0.10
    elif candidate.energy_above_hull > hull_threshold:
        reasons.append("above_hull_threshold")
        score -= min(0.80, candidate.energy_above_hull)
        passed = False
    elif candidate.energy_above_hull <= 0.05:
        reasons.append("stable")
        score += 0.05
    else:
        reasons.append("metastable")
        score -= candidate.energy_above_hull * 0.5

    metallic_hint = candidate.is_metallic is True or (
        candidate.band_gap is not None and candidate.band_gap <= 0.25
    )
    if metallic_hint:
        reasons.append("metallic_or_low_gap")
        score += 0.05
    elif candidate.band_gap is None:
        reasons.append("unknown_gap")
        score -= 0.05
    else:
        reasons.append("wide_gap_penalty")
        score -= min(0.35, candidate.band_gap * 0.10)

    if has_any(elements, METALS) and has_any(elements, NONMETALS):
        reasons.append("mixed_metal_nonmetal_chemistry")
        score += 0.03

    score = max(0.0, min(1.0, score))
    return FilterResult(passed=passed, score=score, reasons=reasons, composition=composition)
