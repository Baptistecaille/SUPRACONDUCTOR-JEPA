"""EBRM-style multi-objective ranking."""

from __future__ import annotations

from typing import Iterable, List, Mapping

from crys_jepa_gnome_sc.chemistry import apply_chemical_filter
from crys_jepa_gnome_sc.graph import build_graph_summary
from crys_jepa_gnome_sc.models import Candidate, RankedCandidate
from crys_jepa_gnome_sc.motif import score_motifs
from crys_jepa_gnome_sc.surprise import score_surprise
from crys_jepa_gnome_sc.tc_model import predict_tc

DEFAULT_WEIGHTS: Mapping[str, float] = {
    "tc": 0.34,
    "filter": 0.18,
    "surprise": 0.16,
    "motif": 0.18,
    "graph": 0.06,
    "synthesizability": 0.08,
}


def synthesizability_score(candidate: Candidate) -> float:
    if candidate.energy_above_hull is None:
        return 0.55
    return round(max(0.0, min(1.0, 1.0 - candidate.energy_above_hull / 0.30)), 3)


def _rank_one(candidate: Candidate, weights: Mapping[str, float]) -> RankedCandidate:
    filter_result = apply_chemical_filter(candidate)
    tc = predict_tc(candidate)
    graph = build_graph_summary(candidate)
    surprise = score_surprise(candidate, filter_result)
    motif = score_motifs(candidate)
    synth = synthesizability_score(candidate)
    tc_norm = min(1.0, tc.kelvin / 180.0)

    rank_score = (
        weights["tc"] * tc_norm
        + weights["filter"] * filter_result.score
        + weights["surprise"] * surprise.score
        + weights["motif"] * motif.score
        + weights["graph"] * graph.periodicity_confidence
        + weights["synthesizability"] * synth
    )
    if not filter_result.passed:
        rank_score *= 0.45

    reasons = (
        filter_result.reasons
        + tc.factors
        + surprise.reasons
        + motif.motifs
    )
    return RankedCandidate(
        rank=0,
        material_id=candidate.material_id,
        formula=candidate.formula,
        passed_filter=filter_result.passed,
        rank_score=round(rank_score, 5),
        predicted_tc_k=tc.kelvin,
        filter_score=filter_result.score,
        surprise_score=surprise.score,
        motif_score=motif.score,
        graph_confidence=graph.periodicity_confidence,
        synthesizability_score=synth,
        reasons=reasons,
    )


def rank_candidates(
    candidates: Iterable[Candidate],
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> List[RankedCandidate]:
    """Rank candidates by weighted, normalized multi-objective score."""

    scored = [_rank_one(candidate, weights) for candidate in candidates]
    ordered = sorted(scored, key=lambda row: row.rank_score, reverse=True)
    return [
        RankedCandidate(
            rank=index,
            material_id=row.material_id,
            formula=row.formula,
            passed_filter=row.passed_filter,
            rank_score=row.rank_score,
            predicted_tc_k=row.predicted_tc_k,
            filter_score=row.filter_score,
            surprise_score=row.surprise_score,
            motif_score=row.motif_score,
            graph_confidence=row.graph_confidence,
            synthesizability_score=row.synthesizability_score,
            reasons=row.reasons,
        )
        for index, row in enumerate(ordered, start=1)
    ]
