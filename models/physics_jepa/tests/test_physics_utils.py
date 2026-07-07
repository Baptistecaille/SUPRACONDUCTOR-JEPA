import torch

from models.physics_jepa.model.physics_utils import (
    allen_dynes_denominator,
    allen_dynes_f1,
    allen_dynes_f2,
    allen_dynes_tc,
    allen_dynes_validity_penalty,
)


def test_allen_dynes_denominator_matches_formula():
    lam = torch.tensor([0.5, 1.0])
    denom = allen_dynes_denominator(lam, mu_star=0.1)
    expected = lam - 0.1 * (1 + 0.62 * lam)
    assert torch.allclose(denom, expected)


def test_allen_dynes_tc_is_positive_and_finite_in_valid_region():
    lam = torch.tensor([0.5, 1.0, 2.0])
    omega = torch.tensor([200.0, 300.0, 400.0])
    tc = allen_dynes_tc(lam, omega)
    assert torch.isfinite(tc).all()
    assert (tc > 0).all()


def test_allen_dynes_tc_increases_with_lambda_holding_omega_fixed():
    omega = torch.tensor([250.0, 250.0, 250.0])
    lam = torch.tensor([0.3, 0.6, 1.0])
    tc = allen_dynes_tc(lam, omega)
    assert tc[0] < tc[1] < tc[2]


def test_validity_penalty_near_zero_deep_in_valid_region():
    # softplus decays exponentially but never hits exactly zero; a deep-valid
    # lambda should still be an order of magnitude below a near-boundary one.
    lam_deep = torch.tensor([5.0])
    lam_boundary = torch.tensor([0.05])
    penalty_deep = allen_dynes_validity_penalty(lam_deep, mu_star=0.1, eps=0.01)
    penalty_boundary = allen_dynes_validity_penalty(lam_boundary, mu_star=0.1, eps=0.01)
    assert penalty_deep.item() < 0.02
    assert penalty_deep.item() < penalty_boundary.item()


def test_validity_penalty_large_when_denominator_invalid():
    lam = torch.tensor([-0.5])
    penalty = allen_dynes_validity_penalty(lam, mu_star=0.1, eps=0.01)
    assert penalty.item() > 0.1


def test_validity_penalty_is_smooth_softplus_not_hard_clamp():
    lam = torch.linspace(-0.2, 0.2, steps=9, requires_grad=True)
    penalty = allen_dynes_validity_penalty(lam, mu_star=0.1, eps=0.01)
    penalty.sum().backward()
    # every point (including deep-invalid ones) has a nonzero, finite gradient --
    # a hard clamp would zero the gradient once past the threshold.
    assert torch.isfinite(lam.grad).all()
    assert (lam.grad.abs() > 1e-8).all()


def test_f1_is_at_least_one_and_increasing_with_lambda():
    # f1 >= 1 always (it only ever pushes Tc up relative to the bare formula),
    # and grows monotonically with lambda.
    lam = torch.tensor([0.0, 0.5, 1.0, 2.0, 5.0])
    f1 = allen_dynes_f1(lam, mu_star=0.1)
    assert (f1 >= 1.0 - 1e-6).all()
    assert torch.all(f1[1:] >= f1[:-1])
    assert torch.isclose(f1[0], torch.tensor(1.0), atol=1e-6)


def test_f1_negative_lambda_clamped_to_one():
    # negative lambda (label-noise tail of this dataset) must not produce a
    # NaN/complex fractional power; f1 clamps lambda to 0 first, giving f1=1.
    lam = torch.tensor([-0.5, -0.05])
    f1 = allen_dynes_f1(lam, mu_star=0.1)
    assert torch.allclose(f1, torch.ones_like(f1))


def test_f2_reduces_to_one_when_omega2_equals_omega_log():
    lam = torch.tensor([0.5, 1.5, 3.0])
    omega_log = torch.tensor([200.0, 300.0, 400.0])
    omega2 = omega_log.clone()
    f2 = allen_dynes_f2(lam, omega_log, omega2, mu_star=0.1)
    assert torch.allclose(f2, torch.ones_like(f2), atol=1e-5)


def test_strong_coupling_correction_increases_tc_relative_to_bare_formula():
    # at lambda >= ~1, f1 > 1, so the corrected Tc must exceed the bare Tc
    # (matching Allen & Dynes' documented high-lambda deviation).
    lam = torch.tensor([1.2, 2.0, 3.0])
    omega = torch.tensor([250.0, 250.0, 250.0])
    tc_bare = allen_dynes_tc(lam, omega, apply_strong_coupling_correction=False)
    tc_corrected = allen_dynes_tc(lam, omega, apply_strong_coupling_correction=True)
    assert (tc_corrected >= tc_bare).all()
    assert (tc_corrected > tc_bare * 1.01).all()


def test_allen_dynes_tc_with_omega2_applies_f2():
    lam = torch.tensor([1.5])
    omega_log = torch.tensor([200.0])
    omega2 = torch.tensor([400.0])  # omega2 != omega_log -> f2 != 1
    tc_no_f2 = allen_dynes_tc(lam, omega_log, apply_strong_coupling_correction=True)
    tc_with_f2 = allen_dynes_tc(
        lam, omega_log, omega2=omega2, apply_strong_coupling_correction=True
    )
    assert not torch.isclose(tc_no_f2, tc_with_f2, atol=1e-4)
    assert torch.isfinite(tc_with_f2).all()
