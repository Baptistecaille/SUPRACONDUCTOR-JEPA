"""Canonical registry of scalar material-property types consumed by property-JEPA.

`consolidated_properties.csv.gz` (see `scripts/consolidate_properties.py` and
`docs/audit/consolidation_report.json`) carries dozens of per-source raw
columns (e.g. `jarvis_optb88vdw_bandgap`, `jarvis_mbj_bandgap`,
`alexandria_band_gap_ind`) that must NOT be silently merged across DFT
functionals/codes for training targets (see
`dft_heterogeneity_warning` in the consolidation report). This module
defines the small set of canonical, physically-distinct *property types*
that `property_jepa.model.PropertyEncoder` / `PropertyPredictor` condition
on -- each canonical type corresponds to one physical quantity; which raw
column feeds a given canonical type for a given row is a data-loading
concern (stage-2 dataset code), not a modelling concern.
"""

from __future__ import annotations

PROPERTY_TYPES: tuple[str, ...] = (
    "formation_energy_peratom",
    "e_above_hull",
    "e_phase_separation",
    "bandgap_optb88vdw",
    "bandgap_mbj",
    "bandgap_hse",
    "dos_ef",
    "magmom_total",
    "bulk_modulus_kv",
    "shear_modulus_gv",
    "poisson_ratio",
    "elastic_tensor_max",
    "tc_experimental",
    "tc_dft_predicted",
)

NUM_PROPERTY_TYPES = len(PROPERTY_TYPES)

PROPERTY_TYPE_INDEX: dict[str, int] = {name: i for i, name in enumerate(PROPERTY_TYPES)}


def property_type_id(name: str) -> int:
    """Return the canonical integer id for a property-type name.

    Raises:
        ValueError: if `name` is not a registered property type.
    """
    try:
        return PROPERTY_TYPE_INDEX[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown property type {name!r}; expected one of {PROPERTY_TYPES}"
        ) from exc
