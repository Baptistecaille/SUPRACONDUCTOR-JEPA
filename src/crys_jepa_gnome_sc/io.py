"""Input and output helpers for candidate tables."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from crys_jepa_gnome_sc.models import Candidate, RankedCandidate


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def candidate_from_mapping(row: Mapping[str, Any]) -> Candidate:
    """Convert a mapping from CSV or JSON into a Candidate."""

    metadata: Dict[str, Any] = {}
    known = {
        "material_id", "formula", "formation_energy_per_atom", "energy_above_hull",
        "band_gap", "density", "space_group", "volume", "num_sites", "is_metallic",
        "metadata",
    }
    for key, value in row.items():
        if key not in known:
            metadata[key] = value
    if isinstance(row.get("metadata"), dict):
        metadata.update(row["metadata"])

    return Candidate(
        material_id=str(row.get("material_id", "")).strip(),
        formula=str(row.get("formula", "")).strip(),
        formation_energy_per_atom=_parse_float(row.get("formation_energy_per_atom")),
        energy_above_hull=_parse_float(row.get("energy_above_hull")),
        band_gap=_parse_float(row.get("band_gap")),
        density=_parse_float(row.get("density")),
        space_group=_parse_int(row.get("space_group")),
        volume=_parse_float(row.get("volume")),
        num_sites=_parse_int(row.get("num_sites")),
        is_metallic=_parse_bool(row.get("is_metallic")),
        metadata=metadata,
    )


def load_candidates(path: str | Path) -> List[Candidate]:
    """Load candidates from CSV or JSON."""

    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        with input_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    elif input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        rows = payload.get("candidates", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON input must be a list or an object with a candidates list")
    else:
        raise ValueError(f"Unsupported input extension: {input_path.suffix}")

    if not rows:
        raise ValueError(f"Input file contains no candidates: {input_path}")
    return [candidate_from_mapping(row) for row in rows]


def _ranked_to_row(candidate: RankedCandidate) -> Dict[str, Any]:
    row = asdict(candidate)
    row["reasons"] = ";".join(candidate.reasons)
    return row


def write_ranked_candidates(candidates: Iterable[RankedCandidate], path: str | Path) -> None:
    """Write ranked candidates to CSV or JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_ranked_to_row(candidate) for candidate in candidates]
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return
    if output_path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported output extension: {output_path.suffix}")
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
