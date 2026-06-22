import csv
import json
import os
import subprocess
import sys

from crys_jepa_gnome_sc.io import load_candidates, write_ranked_candidates
from crys_jepa_gnome_sc.pipeline import screen_candidates


def test_load_candidates_from_csv_and_screen(tmp_path):
    input_path = tmp_path / "candidates.csv"
    input_path.write_text(
        "material_id,formula,energy_above_hull,band_gap,num_sites\n"
        "hydride,LaH10,0.03,0.0,11\n"
        "control,SiO2,0.0,5.5,3\n",
        encoding="utf-8",
    )

    candidates = load_candidates(input_path)
    ranked = screen_candidates(candidates)

    assert len(candidates) == 2
    assert ranked[0].material_id == "hydride"
    assert ranked[0].rank == 1


def test_load_candidates_from_json_object_and_write_json(tmp_path):
    input_path = tmp_path / "candidates.json"
    output_path = tmp_path / "ranked.json"
    input_path.write_text(
        json.dumps({"candidates": [{"material_id": "boride", "formula": "MgB2"}]}),
        encoding="utf-8",
    )

    ranked = screen_candidates(load_candidates(input_path))
    write_ranked_candidates(ranked, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["material_id"] == "boride"
    assert payload[0]["rank"] == 1


def test_cli_screens_csv_to_csv(tmp_path):
    input_path = tmp_path / "candidates.csv"
    output_path = tmp_path / "ranked.csv"
    input_path.write_text(
        "material_id,formula,energy_above_hull,band_gap\n"
        "hydride,LaH10,0.03,0.0\n"
        "boride,MgB2,0.01,0.0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "crys_jepa_gnome_sc",
            "screen",
            str(input_path),
            "--out",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
    )

    assert "Wrote 2 ranked candidates" in result.stdout
    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["material_id"] == "hydride"
    assert rows[0]["rank"] == "1"
