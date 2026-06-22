from crys_jepa_gnome_sc.chemistry import apply_chemical_filter
from crys_jepa_gnome_sc.graph import build_graph_summary
from crys_jepa_gnome_sc.models import Candidate
from crys_jepa_gnome_sc.motif import score_motifs
from crys_jepa_gnome_sc.ranking import rank_candidates
from crys_jepa_gnome_sc.surprise import score_surprise
from crys_jepa_gnome_sc.tc_model import predict_tc


def test_tc_model_rewards_hydride_and_boride_families():
    hydride = Candidate("gnome-LaH10", "LaH10", energy_above_hull=0.03, band_gap=0.0)
    boride = Candidate("gnome-MgB2", "MgB2", energy_above_hull=0.01, band_gap=0.0)
    oxide = Candidate("gnome-SiO2", "SiO2", energy_above_hull=0.0, band_gap=5.5)

    assert predict_tc(hydride).kelvin > predict_tc(boride).kelvin
    assert predict_tc(boride).kelvin > predict_tc(oxide).kelvin
    assert 0.0 <= predict_tc(hydride).confidence <= 1.0


def test_graph_summary_uses_structure_fields_and_formula_fallback():
    structured = Candidate(
        "structured",
        "YBa2Cu3O7",
        density=6.3,
        space_group=123,
        volume=170.0,
        num_sites=13,
    )
    fallback = Candidate("fallback", "MgB2")

    structured_graph = build_graph_summary(structured)
    fallback_graph = build_graph_summary(fallback)

    assert structured_graph.nodes == 13
    assert structured_graph.periodicity_confidence > fallback_graph.periodicity_confidence
    assert fallback_graph.nodes == 3
    assert fallback_graph.edges > 0


def test_surprise_and_motif_scores_are_normalized_and_explainable():
    candidate = Candidate("cuprate", "YBa2Cu3O7", energy_above_hull=0.04, band_gap=0.1)
    filter_result = apply_chemical_filter(candidate)

    surprise = score_surprise(candidate, filter_result)
    motif = score_motifs(candidate)

    assert 0.0 <= surprise.score <= 1.0
    assert surprise.reasons
    assert 0.0 <= motif.score <= 1.0
    assert "cuprate_like" in motif.motifs


def test_ebrm_ranking_orders_promising_candidates_above_controls():
    candidates = [
        Candidate("control", "SiO2", energy_above_hull=0.00, band_gap=5.5),
        Candidate("boride", "MgB2", energy_above_hull=0.01, band_gap=0.0),
        Candidate("hydride", "LaH10", energy_above_hull=0.03, band_gap=0.0),
    ]

    ranked = rank_candidates(candidates)

    assert [row.rank for row in ranked] == [1, 2, 3]
    assert ranked[0].material_id == "hydride"
    assert ranked[-1].material_id == "control"
    assert ranked[0].rank_score > ranked[-1].rank_score
