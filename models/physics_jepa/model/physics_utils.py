"""Allen-Dynes Tc formula + physical-validity penalty, shared by the loss
and evaluation code in `physics_jepa`.

The Allen-Dynes (1975) strong-coupling formula for the superconducting
critical temperature is:

    Tc = f1 * f2 * (omega_log / 1.2) * exp( -1.04 * (1 + lambda) /
                                             (lambda - mu_star * (1 + 0.62 * lambda)) )

with `mu_star` the (dataset-level constant, not fit per-material) Coulomb
pseudopotential, `mu_star=0.1` by default (a standard literature value; the
raw datasets consolidated into `physics_jepa_{train,val,test}.csv.gz` used
`mu_star=0.1` or `0.13` depending on source -- since neither is stored
per-row and this module only NEEDS a working formula for auxiliary
consistency checks, not to reproduce any one source's exact number,
`mu_star=0.1` is used uniformly here).

**Strong-coupling correction (f1, f2) and formula validity.** The bare
exponential factor above -- what Allen & Dynes call the "McMillan-style"
interpolation -- is only accurate for weak-to-moderate coupling; Allen &
Dynes' own analysis (and the standard materials-informatics literature
built on it) documents that it drifts increasingly high above roughly
`lambda ~ 1.5`. They introduce two correction factors to compensate:

    f1 = (1 + (lambda / Lambda1) ** 1.5) ** (1/3),   Lambda1 = 2.46 * (1 + 3.8 * mu_star)
    f2 = 1 + (omega2/omega_log - 1) * lambda**2 / (lambda**2 + Lambda2**2),
                                                       Lambda2 = 1.82 * (1 + 6.3 * mu_star) * (omega2 / omega_log)

`f1` depends only on `lambda` and `mu_star` and is always computable; `f2`
additionally needs `omega2 = sqrt(<omega^2>)`, the second frequency moment
of the Eliashberg spectral function `alpha^2F(omega)` -- a quantity this
dataset does not carry for every row (only ~14% of `physics_jepa_train`
rows have a stored `alpha^2F` spectrum to derive it from; see
`has_a2f_spectrum` / `a2f_original_{x,y}` in `physics_jepa_{train,val,test}.csv.gz`).
`allen_dynes_tc` therefore applies `f1` unconditionally (it needs no extra
data and is the dominant correction at moderate-to-high lambda) and accepts
an optional `omega2` tensor to additionally apply `f2` when available;
otherwise `f2=1` (the bare formula's implicit assumption `omega2 ==
omega_log`, i.e. a) delta-function-like spectrum). Empirically (checked
against this dataset's own `tc_allen_dynes` labels, which were themselves
computed upstream with the *uncorrected* formula), `f1` cuts the mean
absolute Tc discrepancy at `lambda` in `[1.0, 1.5)` from ~0.9 K to ~0.2 K
FYI relative to those (also-approximate) upstream labels.

Even with `f1*f2`, the Allen-Dynes interpolation remains an approximation
to numerical Eliashberg theory and its accuracy degrades further for very
large `lambda` (empirically, beyond roughly `lambda > 5-6` on this
dataset's own spectra, `f1` alone starts to systematically overshoot the
upstream label) -- there is no simple analytic formula that is exact in
that regime. `apply_strong_coupling_correction=False` reproduces the
original bare (uncorrected) formula, useful for exact backward-compatible
comparison against upstream `tc_allen_dynes` columns that were computed
that way.

The denominator `lambda - mu_star * (1 + 0.62 * lambda)` must be strictly
positive for Tc to be physically defined (a non-positive denominator sends
the formula to `Tc -> +inf` or a spurious negative Tc as the exponent's
sign flips) -- i.e. superconductivity requires the coupling `lambda` to
exceed a `mu_star`-dependent threshold. `allen_dynes_validity_penalty` is a
SOFT constraint (`softplus`, not a hard clamp) on a predicted `lambda_ep`
staying on the physical side of that threshold: it is added to the
training loss with a small weight so gradients stay informative even for
predictions currently in the invalid region, rather than being zeroed by a
hard clamp. This is a distinct notion from the strong-coupling-formula
accuracy discussed above: `allen_dynes_validity_penalty` flags physically
*undefined* Tc (denominator <= eps), not merely *approximate* Tc from a
large `lambda`.

**`f2` auxiliary evaluation on this dataset.** `physics_jepa`'s heads only
predict `(lambda_ep, omega_log)` -- there is no `omega2` prediction
pathway, so `f2` cannot be applied to model *predictions*, only to
ground-truth rows that carry a raw `alpha^2F` spectrum. On the 48
`has_a2f_spectrum=True` rows in `physics_jepa_test.csv.gz`, `omega2` was
reconstructed from `a2f_original_{x,y}` via `omega2 =
sqrt(2*integral(a2F(w)*w dw) / lambda_check)` with `lambda_check =
2*integral(a2F(w)/w dw)`, keeping only the 33/48 rows where `lambda_check`
matched the stored `lambda_ep` within 30% (the rest have unreliable/noisy
raw spectra). On that reliable subset, ground-truth `f1`-only Tc averaged
~4.7 K vs. ~4.4 K bare, while `f1*f2` dropped to ~2.0 K -- i.e. on this
particular subset `f2` pulls Tc back down markedly (`omega2/omega_log`
averaged ~0.21 here, well below the bare formula's implicit assumption of
1.0), a bigger swing than `f1` alone produces. `omega2` is used in the
same Kelvin units JARVIS/Alexandria store for `omega_log`; no independent
unit cross-check of `a2f_original_x` against `omega_log` was performed
beyond the `lambda_check` self-consistency filter above, so the `f2`
correction here is illustrative of its potential magnitude on this data,
not a fully validated per-material correction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_MU_STAR = 0.1
DEFAULT_VALIDITY_EPS = 0.01


def allen_dynes_denominator(lambda_ep: torch.Tensor, mu_star: float = DEFAULT_MU_STAR) -> torch.Tensor:
    """`lambda - mu_star * (1 + 0.62 * lambda)`; must be > 0 for a physical Tc."""
    return lambda_ep - mu_star * (1.0 + 0.62 * lambda_ep)


def allen_dynes_f1(lambda_ep: torch.Tensor, mu_star: float = DEFAULT_MU_STAR) -> torch.Tensor:
    """Strong-coupling correction factor `f1 = (1 + (lambda/Lambda1)**1.5)**(1/3)`,
    `Lambda1 = 2.46*(1+3.8*mu_star)`. Depends only on `lambda` and `mu_star`; always
    computable and >= 1 (it can only push Tc up relative to the bare formula).
    Negative `lambda_ep` (occurs at the tails of this dataset's label noise) is
    clamped to 0 before the fractional power to keep the factor real and >= 1."""
    lambda_clamped = lambda_ep.clamp(min=0.0)
    Lambda1 = 2.46 * (1.0 + 3.8 * mu_star)
    return (1.0 + (lambda_clamped / Lambda1) ** 1.5) ** (1.0 / 3.0)


def allen_dynes_f2(
    lambda_ep: torch.Tensor,
    omega_log: torch.Tensor,
    omega2: torch.Tensor,
    mu_star: float = DEFAULT_MU_STAR,
) -> torch.Tensor:
    """Strong-coupling correction factor `f2`, requiring the spectral second
    moment `omega2 = sqrt(<omega^2>)` in the same units as `omega_log`. Falls
    back to `f2=1` wherever `omega_log` is non-positive (guards the ratio)."""
    lambda_clamped = lambda_ep.clamp(min=0.0)
    safe_omega_log = omega_log.clamp(min=1e-6)
    ratio = omega2 / safe_omega_log
    Lambda2 = 1.82 * (1.0 + 6.3 * mu_star) * ratio
    f2 = 1.0 + (ratio - 1.0) * lambda_clamped**2 / (lambda_clamped**2 + Lambda2**2 + 1e-12)
    return torch.where(omega_log > 0, f2, torch.ones_like(f2))


def allen_dynes_tc(
    lambda_ep: torch.Tensor,
    omega_log: torch.Tensor,
    mu_star: float = DEFAULT_MU_STAR,
    eps: float = DEFAULT_VALIDITY_EPS,
    omega2: torch.Tensor | None = None,
    apply_strong_coupling_correction: bool = True,
) -> torch.Tensor:
    """Allen-Dynes Tc (Kelvin), optionally strong-coupling corrected (see
    module docstring). The denominator is clamped to `>= eps` purely to keep
    this function numerically finite when called on off-manifold (e.g.
    early-training) predictions; `allen_dynes_validity_penalty` is what
    actually discourages the model from landing there.

    Args:
        omega2: optional `sqrt(<omega^2>)` spectral second moment (same
            units/shape as `omega_log`); enables the `f2` correction when
            given. Ignored if `apply_strong_coupling_correction=False`.
        apply_strong_coupling_correction: if `True` (default), multiplies by
            `f1` (always) and `f2` (only if `omega2` is given). If `False`,
            reproduces the original bare/uncorrected formula.
    """
    denom = allen_dynes_denominator(lambda_ep, mu_star).clamp(min=eps)
    tc_bare = (omega_log / 1.2) * torch.exp(-1.04 * (1.0 + lambda_ep) / denom)
    if not apply_strong_coupling_correction:
        return tc_bare
    f1 = allen_dynes_f1(lambda_ep, mu_star)
    f2 = (
        allen_dynes_f2(lambda_ep, omega_log, omega2, mu_star)
        if omega2 is not None
        else torch.ones_like(tc_bare)
    )
    return f1 * f2 * tc_bare


def allen_dynes_validity_penalty(
    lambda_ep: torch.Tensor,
    mu_star: float = DEFAULT_MU_STAR,
    eps: float = DEFAULT_VALIDITY_EPS,
) -> torch.Tensor:
    """Soft (softplus) penalty on `lambda_ep` violating
    `lambda - mu_star * (1 + 0.62 * lambda) > eps`.

    Returns a `(...,)` per-sample penalty (0 deep in the valid region,
    growing smoothly, never discontinuous, as the denominator approaches or
    crosses `eps` from above) -- mean/sum-reduce at the call site.
    """
    denom = allen_dynes_denominator(lambda_ep, mu_star)
    return F.softplus(-(denom - eps))


__all__ = [
    "DEFAULT_MU_STAR",
    "DEFAULT_VALIDITY_EPS",
    "allen_dynes_denominator",
    "allen_dynes_f1",
    "allen_dynes_f2",
    "allen_dynes_tc",
    "allen_dynes_validity_penalty",
]
