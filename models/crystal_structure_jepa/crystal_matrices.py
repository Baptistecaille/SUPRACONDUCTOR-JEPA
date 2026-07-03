import torch
import torch.nn.functional as F
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pathlib import Path
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


def compute_lattice_6d(lattice_matrix: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Polar SVD decomposition of a 3×3 lattice matrix → 6D upper-triangular representation.

    Computes the symmetric positive-definite factor P' = U·P·Uᵀ where
    W,S,Vᵀ = SVD(L), U = W·Vᵀ, P = V·diag(S²)·Vᵀ, then extracts the
    6 upper-triangular elements and normalises by num_atoms^(1/3).

    Args:
        lattice_matrix: Lattice vectors as a (3, 3) float tensor (rows = a, b, c).
        num_atoms: Number of atoms in the unit cell, used for normalisation.

    Returns:
        Tensor of shape (6,) — upper triangle of P', scaled by num_atoms^(-1/3).
    """
    W, S, Vt = torch.linalg.svd(lattice_matrix)
    V = Vt.T
    U = W @ Vt
    P = V @ torch.diag(S ** 2) @ V.T
    P_prime = U @ P @ U.T
    P_prime[torch.abs(P_prime) < 1e-5] = 0.0
    tri = torch.triu_indices(3, 3)
    lattice_6d = P_prime[tri[0], tri[1]]
    return lattice_6d / (num_atoms ** (1 / 3))


def _build_atomic_one_hot(atomic_numbers: torch.Tensor) -> torch.Tensor:
    """Convert 1-indexed atomic numbers H=1..100 to a 100-class one-hot matrix."""
    return F.one_hot((atomic_numbers - 1).long(), num_classes=100).float()


def build_encoder1_atom_feature_matrix(
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    """Build Encoder 1 per-atom features from the corrupted crystal.

    Each row is vᵢ = [ Xᵢ (3) ∥ one_hot(Aᵢ, 100) ], shape (N, 103).
    """
    one_hot = _build_atomic_one_hot(atomic_numbers)
    return torch.cat([frac_coords.float(), one_hot], dim=-1)


def build_encoder2_atom_feature_matrix(
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    family_one_hot: torch.Tensor,
) -> torch.Tensor:
    """Build Encoder 2 per-atom features from the full EMA crystal.

    Each row is vᵢ = [ Xᵢ (3) ∥ one_hot(Aᵢ, 100) ∥ family_one_hot (7) ],
    shape (N, 110).
    """
    N = frac_coords.shape[0]
    base_features = build_encoder1_atom_feature_matrix(frac_coords, atomic_numbers)
    family_expanded = family_one_hot.float().unsqueeze(0).expand(N, -1)
    return torch.cat([base_features, family_expanded], dim=-1)


def build_atom_feature_matrix(
    frac_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    lattice_6d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backward-compatible alias for Encoder 1 atom features.

    The `lattice_6d` argument is ignored intentionally: lattice values are no
    longer part of the JEPA token representation.
    """
    return build_encoder1_atom_feature_matrix(frac_coords, atomic_numbers)


def _close(left: float, right: float, tol: float) -> bool:
    return abs(left - right) <= tol


def crystal_family_one_hot(
    lattice: Lattice,
    length_tol: float = 1e-3,
    angle_tol: float = 1e-2,
) -> torch.Tensor:
    """Classify a lattice into the seven crystal systems as a one-hot vector."""
    a, b, c = lattice.abc
    alpha, beta, gamma = lattice.angles

    eq_ab = _close(a, b, length_tol)
    eq_bc = _close(b, c, length_tol)
    alpha90 = _close(alpha, 90.0, angle_tol)
    beta90 = _close(beta, 90.0, angle_tol)
    gamma90 = _close(gamma, 90.0, angle_tol)
    all_90 = alpha90 and beta90 and gamma90

    index = 0
    if eq_ab and eq_bc and all_90:
        index = 6
    elif eq_ab and alpha90 and beta90 and _close(gamma, 120.0, angle_tol):
        index = 5
    elif (
        eq_ab
        and eq_bc
        and _close(alpha, beta, angle_tol)
        and _close(beta, gamma, angle_tol)
    ):
        index = 4
    elif eq_ab and all_90:
        index = 3
    elif all_90:
        index = 2
    elif alpha90 and gamma90:
        index = 1

    return F.one_hot(torch.tensor(index), num_classes=7).float()


def spatial_block_visible_mask(
    frac_coords: torch.Tensor,
    center: torch.Tensor,
    box_size: float | torch.Tensor,
) -> torch.Tensor:
    """Return True for atoms outside a toric fractional-coordinate box."""
    center = center.to(device=frac_coords.device, dtype=frac_coords.dtype)
    size = torch.as_tensor(box_size, device=frac_coords.device, dtype=frac_coords.dtype)
    toric_delta = torch.remainder(frac_coords - center + 0.5, 1.0) - 0.5
    inside = torch.abs(toric_delta).le(size / 2).all(dim=-1)
    return ~inside


def sample_spatial_block(
    frac_coords: torch.Tensor,
    volume_range: tuple[float, float] = (0.2, 0.5),
    center: torch.Tensor | None = None,
    box_size: float | torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a spatial block and return `(visible_mask, mask_box)`.

    `mask_box` is `[center_x, center_y, center_z, box_size]`, where `box_size`
    is a cube side length in fractional coordinates. When all atoms are masked,
    the atom farthest from the sampled center is kept visible. When no atom is
    masked and at least two atoms exist, the atom closest to the center is masked.
    """
    device = frac_coords.device
    dtype = frac_coords.dtype
    if center is None:
        center = torch.rand(3, device=device, dtype=dtype, generator=generator)
    else:
        center = center.to(device=device, dtype=dtype)

    if box_size is None:
        volume = torch.empty((), device=device, dtype=dtype).uniform_(
            volume_range[0], volume_range[1], generator=generator
        )
        size = volume.pow(torch.tensor(1.0 / 3.0, device=device, dtype=dtype))
    else:
        size = torch.as_tensor(box_size, device=device, dtype=dtype)

    visible = spatial_block_visible_mask(frac_coords, center, size)
    if frac_coords.numel() > 0 and not visible.any():
        toric_delta = torch.remainder(frac_coords - center + 0.5, 1.0) - 0.5
        farthest_atom = torch.norm(toric_delta, dim=-1).argmax()
        visible[farthest_atom] = True

    if frac_coords.shape[0] >= 2 and visible.all():
        toric_delta = torch.remainder(frac_coords - center + 0.5, 1.0) - 0.5
        closest_atom = torch.norm(toric_delta, dim=-1).argmin()
        visible[closest_atom] = False

    mask_box = torch.cat([center, size.reshape(1)])
    return visible, mask_box


def structure_to_tensors(
    structure: Structure,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Convert a pymatgen Structure into raw PyTorch tensors for Crys-JEPA.

    Atoms are sorted deterministically by (atomic_number, x, y, z) so that
    two representations of the same crystal always produce the same ordering.
    The lattice is reduced to a 7-class crystal-family one-hot vector.

    Args:
        structure: A pymatgen Structure (primitive or reduced cell recommended).

    Returns:
        frac_coords: Fractional coordinates, shape (N, 3), float32.
        atomic_numbers: 1-indexed atomic numbers, shape (N,), long.
        family_one_hot: Crystal family one-hot vector, shape (7,), float32.
        num_atoms: Number of atoms N (int).
    """
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)
    atomic_numbers = torch.tensor(
        [
            max(site.species.items(), key=lambda item: (float(item[1]), item[0].Z))[0].Z
            for site in structure
        ],
        dtype=torch.long,
    )
    num_atoms = len(atomic_numbers)

    # Sort atoms deterministically by (atomic_number, x, y, z)
    sort_key = (
        1000 * atomic_numbers.float()
        + 100 * frac_coords[:, 0]
        + 10 * frac_coords[:, 1]
        + frac_coords[:, 2]
    )
    order = torch.argsort(sort_key)
    frac_coords = frac_coords[order]
    atomic_numbers = atomic_numbers[order]

    family_one_hot = crystal_family_one_hot(structure.lattice)

    return frac_coords, atomic_numbers, family_one_hot, num_atoms


def build_energy_weight_matrix(ef_per_atom: torch.Tensor) -> torch.Tensor:
    """Build the B×B energy weighting matrix Ω for the InfoNCE loss (Crys-JEPA §3.2, Eq. 8).

    Off-diagonal entries repel embeddings of crystals whose formation energies
    differ strongly; the diagonal is set to 1 to preserve the self-similarity term:
      Ωᵢₖ = 1 − exp(−|Eᶠᵢ − Eᶠₖ|)   for i ≠ k
      Ωᵢᵢ = 1

    Args:
        ef_per_atom: Formation energy per atom for each crystal in the batch,
                     shape (B,). Units should be consistent (e.g. eV/atom).

    Returns:
        Symmetric tensor of shape (B, B) with values in (0, 1] on off-diagonal
        and exactly 1 on the diagonal.
    """
    diff = torch.abs(ef_per_atom.unsqueeze(0) - ef_per_atom.unsqueeze(1))
    weight = 1.0 - torch.exp(-diff)
    diag = torch.eye(len(ef_per_atom), dtype=torch.float32, device=ef_per_atom.device)
    return weight + diag


class SuperconductorDataset(Dataset):
    """Dataset backed by CIF strings and formation energies in data/jepa/mp.csv.gz."""

    def __init__(
        self,
        csv_path: str | Path = Path("data/jepa/mp.csv.gz"),
        deduplicate: bool = True,
    ):
        self.csv_path = Path(csv_path)
        self.data = self._load_data(self.csv_path)

        required_columns = {"material_id", "cif", "ef_per_atom"}
        missing_columns = required_columns.difference(self.data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{self.csv_path} is missing columns: {missing}")

        if deduplicate:
            self.data = self._deduplicate_data(self.data)

    @staticmethod
    def _load_data(csv_path: Path) -> pd.DataFrame:
        data = pd.read_csv(csv_path)
        if {"material_id", "cif", "ef_per_atom"}.issubset(data.columns):
            return data

        raw_data = pd.read_csv(csv_path, skiprows=1)
        raw_columns = {
            "material_id_2",
            "cif",
            "formation_energy_per_atom_2",
        }
        if not raw_columns.issubset(raw_data.columns):
            return data

        normalized = pd.DataFrame(
            {
                "material_id": raw_data["material_id_2"],
                "cif": raw_data["cif"].map(
                    lambda value: SuperconductorDataset._read_cif_value(csv_path, value)
                ),
                "ef_per_atom": raw_data["formation_energy_per_atom_2"],
            }
        )
        if "tc" in raw_data.columns:
            normalized["tc"] = raw_data["tc"]
        return normalized.dropna(subset=["material_id", "cif", "ef_per_atom"]).reset_index(
            drop=True
        )

    @staticmethod
    def _read_cif_value(csv_path: Path, value: object) -> str | None:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if text.startswith("data_") or "\n" in text:
            return text

        candidates = [
            Path(text),
            csv_path.parent / Path(text),
            csv_path.parent / "cifs" / Path(text).name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.read_text()
        return None

    @staticmethod
    def _deduplicate_data(data: pd.DataFrame) -> pd.DataFrame:
        """Keep one row per material when repeated exports duplicate structures."""

        return data.drop_duplicates(subset=["material_id"], keep="first").reset_index(
            drop=True
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.data.iloc[idx]
        structure = Structure.from_str(row["cif"], fmt="cif")
        frac_coords, atomic_numbers, family_one_hot, num_atoms = structure_to_tensors(
            structure
        )
        encoder2_atom_features = build_encoder2_atom_feature_matrix(
            frac_coords, atomic_numbers, family_one_hot
        )

        sample = {
            "material_id": row["material_id"],
            "frac_coords": frac_coords,
            "atomic_numbers": atomic_numbers,
            "family_one_hot": family_one_hot,
            "encoder2_atom_features": encoder2_atom_features,
            "ef_per_atom": torch.tensor(row["ef_per_atom"], dtype=torch.float32),
            "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
        }
        if "tc" in self.data.columns:
            sample["tc"] = torch.tensor(row["tc"], dtype=torch.float32)
        return sample


def collate_crystals(
    samples: list[dict[str, torch.Tensor | str]],
) -> dict[str, torch.Tensor | list[str]]:
    """Pad full and spatially masked crystal atom matrices into a dense mini-batch."""

    encoder1_atom_features = []
    encoder1_num_atoms = []
    mask_boxes = []

    for sample in samples:
        frac_coords = sample["frac_coords"]
        atomic_numbers = sample["atomic_numbers"]
        visible, mask_box = sample_spatial_block(frac_coords)
        encoder1_atom_features.append(
            build_encoder1_atom_feature_matrix(
                frac_coords[visible], atomic_numbers[visible]
            )
        )
        encoder1_num_atoms.append(
            torch.tensor(int(visible.sum().item()), dtype=torch.long)
        )
        mask_boxes.append(mask_box)

    encoder2_atom_features = [sample["encoder2_atom_features"] for sample in samples]
    num_atoms = torch.stack([sample["num_atoms"] for sample in samples])
    encoder1_num_atoms = torch.stack(encoder1_num_atoms)

    padded_encoder1 = pad_sequence(encoder1_atom_features, batch_first=True)
    padded_encoder2 = pad_sequence(encoder2_atom_features, batch_first=True)
    max_encoder1_atoms = int(encoder1_num_atoms.max().item())
    max_encoder2_atoms = int(num_atoms.max().item())
    encoder1_atom_mask = (
        torch.arange(max_encoder1_atoms).unsqueeze(0) < encoder1_num_atoms.unsqueeze(1)
    )
    encoder2_atom_mask = (
        torch.arange(max_encoder2_atoms).unsqueeze(0) < num_atoms.unsqueeze(1)
    )

    batch = {
        "material_id": [str(sample["material_id"]) for sample in samples],
        "encoder1_atom_features": padded_encoder1,
        "encoder1_atom_mask": encoder1_atom_mask,
        "encoder1_num_atoms": encoder1_num_atoms,
        "encoder2_atom_features": padded_encoder2,
        "encoder2_atom_mask": encoder2_atom_mask,
        "family_one_hot": torch.stack(
            [sample["family_one_hot"] for sample in samples]
        ),
        "mask_box": torch.stack(mask_boxes),
        "ef_per_atom": torch.stack([sample["ef_per_atom"] for sample in samples]),
        "num_atoms": num_atoms,
    }
    if "tc" in samples[0]:
        batch["tc"] = torch.stack([sample["tc"] for sample in samples])
    return batch


def create_dataloader(
    csv_path: str | Path = Path("data/jepa/mp.csv.gz"),
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    deduplicate: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Load the dataset and create a DataLoader with the data folder dataset."""

    dataset = SuperconductorDataset(csv_path, deduplicate=deduplicate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_crystals,
        drop_last=drop_last,
    )
