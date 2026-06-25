"""
scripts/prepare_data.py — Téléchargement et préparation des données

Sources :
  - SuperCon database (via matminer) : ~16 000 supraconducteurs avec Tc
  - Materials Project (via mp-api)   : structures non-SC comme négatives

Génère :
  data/supercon_structures.json   — {structure, label:1, tc}
  data/nonsc_structures.json      — {structure, label:0}

Usage :
  pip install matminer mp-api pymatgen spglib
  python scripts/prepare_data.py --mp-api-key <VOTRE_CLE>

La clé API Materials Project est gratuite sur https://materialsproject.org
"""

import os
import json
import hashlib
import argparse
import random


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_matminer_dataset_once(name: str) -> str:
    from matminer.datasets.dataset_retrieval import get_dataset_attribute
    from matminer.datasets.utils import _get_data_home
    import requests

    file_type = get_dataset_attribute(name, "file_type")
    url = get_dataset_attribute(name, "url")
    expected_hash = get_dataset_attribute(name, "hash")
    data_home = _get_data_home()
    os.makedirs(data_home, exist_ok=True)
    cache_path = os.path.join(data_home, f"{name}.{file_type}")

    if os.path.exists(cache_path) and os.path.getsize(cache_path) == 0:
        os.remove(cache_path)

    urls = [url]
    if "figshare.com/ndownloader/files/" in url:
        file_id = url.rstrip("/").split("/")[-1]
        urls.append(f"https://api.figshare.com/v2/file/download/{file_id}")

    if not os.path.exists(cache_path):
        last_error = None
        for download_url in urls:
            try:
                print(f"Téléchargement {name} depuis {download_url}...")
                response = requests.get(download_url, timeout=60)
                response.raise_for_status()
                if not response.content:
                    raise RuntimeError(
                        f"{download_url} a renvoyé 0 octet."
                    )
                with open(cache_path, "wb") as f:
                    f.write(response.content)
                break
            except Exception as exc:
                last_error = exc
                if os.path.exists(cache_path):
                    os.remove(cache_path)
        else:
            raise RuntimeError(
                "Aucune URL de téléchargement n'a fonctionné. "
                f"Dernière erreur: {last_error}"
            )

    actual_hash = _sha256(cache_path)
    if actual_hash != expected_hash:
        os.remove(cache_path)
        raise RuntimeError(
            f"Hash invalide pour {name}: {actual_hash}. "
            "Le fichier reçu ne correspond pas à la metadata matminer."
        )

    return cache_path


def load_supercon_table(supercon_csv: str | None = None):
    """Charge SuperCon avec les colonnes normalisées formula/tc."""
    if supercon_csv:
        import pandas as pd

        df = pd.read_csv(supercon_csv)
    else:
        from matminer.datasets import load_dataset

        try:
            _download_matminer_dataset_once("superconductivity2018")
            df = load_dataset(
                "superconductivity2018",
                download_if_missing=False,
            )  # composition + Tc
        except Exception as exc:
            raise RuntimeError(
                "Matminer n'a pas pu télécharger superconductivity2018 : "
                f"{exc}. "
                "Relance la commande plus tard, ou fournis un CSV local avec "
                "`--supercon-csv chemin.csv` contenant les colonnes "
                "`composition` et `Tc`."
            ) from exc

    rename_map = {}
    if "composition" in df.columns:
        rename_map["composition"] = "formula"
    if "critical_temp" in df.columns:
        rename_map["critical_temp"] = "tc"
    if "Tc" in df.columns:
        rename_map["Tc"] = "tc"
    df = df.rename(columns=rename_map)

    missing = {"formula", "tc"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes SuperCon manquantes: {sorted(missing)}. "
            "Attendu: composition/Tc ou formula/tc."
        )

    df = df[["formula", "tc"]].dropna()
    df = df[df["tc"] > 0].sort_values("tc", ascending=False)
    df = df.drop_duplicates(subset="formula", keep="first")
    return df


def _mp_formula(formula: str) -> str | None:
    """Convertit une formule SuperCon en formule réduite compatible MP."""
    from pymatgen.core import Composition

    try:
        return Composition(formula).reduced_formula
    except Exception:
        return None


def _mp_formula_candidates(formula: str) -> list[str]:
    """Formules candidates: exacte puis version arrondie pour MP."""
    from pymatgen.core import Composition

    try:
        composition = Composition(formula)
    except Exception:
        return []

    candidates = [composition.reduced_formula]

    rounded = {}
    for el, amount in composition.get_el_amt_dict().items():
        rounded_amount = int(round(amount))
        if rounded_amount > 0:
            rounded[el] = rounded_amount
    if rounded:
        try:
            candidates.append(Composition(rounded).reduced_formula)
        except Exception:
            pass

    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def get_supercon_structures(
    mp_api_key: str,
    output_path: str,
    max_samples: int = 5000,
    supercon_csv: str | None = None,
) -> set[str]:
    """Télécharge les structures des supraconducteurs depuis SuperCon + MP."""
    from mp_api.client import MPRester

    print("Chargement de la base SuperCon (matminer)...")
    df = load_supercon_table(supercon_csv)
    df["mp_formula_candidates"] = df["formula"].map(_mp_formula_candidates)
    df = df[df["mp_formula_candidates"].map(bool)]
    print(f"  {len(df)} supraconducteurs avec Tc > 0")

    records = []
    positive_ids = set()
    formula_to_row = {}
    candidate_order = []
    for row in df.to_dict("records"):
        for candidate in row["mp_formula_candidates"]:
            if candidate not in formula_to_row:
                formula_to_row[candidate] = row
                candidate_order.append(candidate)

    with MPRester(mp_api_key) as mpr:
        print(
            f"Recherche MP par lots jusqu'à {max_samples} structures SC "
            f"({len(candidate_order)} formules candidates)..."
        )
        batch_size = 100
        for start in range(0, len(candidate_order), batch_size):
            if len(records) >= max_samples:
                break
            batch = candidate_order[start: start + batch_size]
            try:
                docs = mpr.materials.summary.search(
                    formula=batch,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "structure",
                        "energy_above_hull",
                    ],
                    num_chunks=1,
                    chunk_size=1000,
                )
            except Exception as e:
                print(f"  ⚠ lot {start // batch_size + 1} ignoré : {e}")
                continue

            best_doc_by_formula = {}
            for doc in docs:
                doc_formula = _mp_formula(doc.formula_pretty)
                if doc_formula not in formula_to_row:
                    continue
                previous = best_doc_by_formula.get(doc_formula)
                if previous is None or float(doc.energy_above_hull or 999.0) < float(
                    previous.energy_above_hull or 999.0
                ):
                    best_doc_by_formula[doc_formula] = doc

            for formula in batch:
                if len(records) >= max_samples:
                    break
                doc = best_doc_by_formula.get(formula)
                if doc is None or str(doc.material_id) in positive_ids:
                    continue
                row = formula_to_row[formula]
                positive_ids.add(str(doc.material_id))
                records.append(
                    {
                        "material_id": str(doc.material_id),
                        "formula": row["formula"],
                        "mp_formula": formula,
                        "mp_formula_pretty": doc.formula_pretty,
                        "structure": doc.structure.as_dict(),
                        "label": 1,
                        "tc": float(row["tc"]),
                    }
                )

            print(
                f"  {min(start + batch_size, len(candidate_order))}/"
                f"{len(candidate_order)} candidates testées, "
                f"{len(records)}/{max_samples} structures SC récupérées"
            )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(records, f)
    print(f"✓ {len(records)} structures SC sauvegardées → {output_path}")
    return positive_ids


def get_nonsc_structures(
    mp_api_key: str,
    output_path: str,
    n_samples: int = 15000,
    exclude_material_ids: set[str] | None = None,
):
    """Télécharge des structures non-SC depuis Materials Project.

    Critère : pas de Tc dans SuperCon, E_above_hull < 0.05 eV/atom (stables).
    """
    from mp_api.client import MPRester

    exclude_material_ids = exclude_material_ids or set()
    print(f"Téléchargement de {n_samples} structures non-SC depuis MP...")
    records = []
    with MPRester(mp_api_key) as mpr:
        chunk_size = 500
        query_limit = max(n_samples * 2, n_samples + len(exclude_material_ids), chunk_size)
        num_chunks = max(1, (query_limit + chunk_size - 1) // chunk_size)
        results = mpr.materials.summary.search(
            energy_above_hull=(0, 0.05),
            fields=["material_id", "structure"],
            num_chunks=num_chunks,
            chunk_size=chunk_size,
        )
        random.shuffle(results)
        for doc in results[:query_limit]:
            if len(records) >= n_samples:
                break
            if str(doc.material_id) in exclude_material_ids:
                continue
            try:
                records.append({
                    "material_id": str(doc.material_id),
                    "structure":   doc.structure.as_dict(),
                    "label":       0,
                    "tc":          0.0,
                })
            except Exception:
                pass
            if len(records) % 1000 == 0:
                print(f"  {len(records)}/{n_samples} structures non-SC")

    with open(output_path, "w") as f:
        json.dump(records, f)
    print(f"✓ {len(records)} structures non-SC sauvegardées → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp-api-key", required=True, help="Clé API Materials Project")
    parser.add_argument("--data-dir",   default="data")
    parser.add_argument("--max-sc",     type=int, default=5000)
    parser.add_argument("--max-nonsc",  type=int, default=15000)
    parser.add_argument(
        "--supercon-csv",
        default=None,
        help="CSV local SuperCon avec colonnes composition/Tc ou formula/tc",
    )
    args = parser.parse_args()

    positive_ids = get_supercon_structures(
        args.mp_api_key,
        os.path.join(args.data_dir, "supercon_structures.json"),
        args.max_sc,
        args.supercon_csv,
    )
    get_nonsc_structures(
        args.mp_api_key,
        os.path.join(args.data_dir, "nonsc_structures.json"),
        args.max_nonsc,
        positive_ids,
    )
    print("\nDonnées prêtes. Lance maintenant :")
    print("  python pretrain.py")
    print("  python finetune.py")


if __name__ == "__main__":
    main()
