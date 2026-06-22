from crys_jepa_gnome_sc.chemistry import apply_chemical_filter, parse_formula
from crys_jepa_gnome_sc.models import Candidate


def test_parse_formula_handles_common_superconductor_formulas():
    assert parse_formula("MgB2").amounts == {"Mg": 1.0, "B": 2.0}
    assert parse_formula("LaH10").amounts == {"La": 1.0, "H": 10.0}
    assert parse_formula("YBa2Cu3O7").amounts == {
        "Y": 1.0,
        "Ba": 2.0,
        "Cu": 3.0,
        "O": 7.0,
    }


def test_parse_formula_rejects_invalid_inputs():
    assert not parse_formula("").valid
    assert not parse_formula("123").valid
    assert not parse_formula("Xx2O").valid


def test_chemical_filter_keeps_stable_low_gap_candidates():
    candidate = Candidate(
        material_id="mp-test",
        formula="MgB2",
        energy_above_hull=0.02,
        band_gap=0.0,
    )

    result = apply_chemical_filter(candidate)

    assert result.passed
    assert result.score > 0.8
    assert "stable" in result.reasons


def test_chemical_filter_rejects_radioactive_and_unstable_candidates():
    radioactive = Candidate(material_id="bad-1", formula="UO2", energy_above_hull=0.01)
    unstable = Candidate(material_id="bad-2", formula="MgB2", energy_above_hull=0.45)

    radioactive_result = apply_chemical_filter(radioactive)
    unstable_result = apply_chemical_filter(unstable)

    assert not radioactive_result.passed
    assert "radioactive_or_unsupported_element" in radioactive_result.reasons
    assert not unstable_result.passed
    assert "above_hull_threshold" in unstable_result.reasons
