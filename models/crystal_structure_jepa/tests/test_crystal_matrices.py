import torch
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from crystal_matrices import (
    SuperconductorDataset,
    build_atom_feature_matrix,
    build_encoder1_atom_feature_matrix,
    build_encoder2_atom_feature_matrix,
    build_energy_weight_matrix,
    collate_crystals,
    compute_lattice_6d,
    create_dataloader,
    crystal_family_one_hot,
    sample_spatial_block,
    spatial_block_visible_mask,
    structure_to_tensors,
)


def test_compute_lattice_6d_shape():
    L = torch.eye(3, dtype=torch.float32)
    out = compute_lattice_6d(L, num_atoms=4)
    assert out.shape == (6,)


def test_compute_lattice_6d_identity():
    L = torch.eye(3, dtype=torch.float32)
    out = compute_lattice_6d(L, num_atoms=4)
    scale = 4 ** (1 / 3)
    expected = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0]) / scale
    assert torch.allclose(out, expected, atol=1e-5)


def test_compute_lattice_6d_normalization():
    L = torch.eye(3, dtype=torch.float32)
    out8 = compute_lattice_6d(L, num_atoms=8)
    out1 = compute_lattice_6d(L, num_atoms=1)
    # Check that non-zero elements are smaller for larger num_atoms
    nonzero_mask = out1 != 0
    assert (out8[nonzero_mask] < out1[nonzero_mask]).all()


def test_build_atom_feature_matrix_shape():
    N = 5
    frac_coords = torch.rand(N, 3)
    atomic_numbers = torch.tensor([1, 6, 8, 14, 26], dtype=torch.long)
    out = build_encoder1_atom_feature_matrix(frac_coords, atomic_numbers)
    assert out.shape == (N, 103)


def test_build_encoder2_atom_feature_matrix_shape():
    N = 5
    frac_coords = torch.rand(N, 3)
    atomic_numbers = torch.tensor([1, 6, 8, 14, 26], dtype=torch.long)
    family_one_hot = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)
    out = build_encoder2_atom_feature_matrix(
        frac_coords, atomic_numbers, family_one_hot
    )
    assert out.shape == (N, 110)
    assert torch.allclose(out[:, 103:], family_one_hot.expand(N, -1))


def test_build_atom_feature_matrix_one_hot():
    # Carbon (atomic_number=6) → one_hot index 5 should be 1
    frac_coords = torch.rand(1, 3)
    atomic_numbers = torch.tensor([6], dtype=torch.long)
    out = build_encoder1_atom_feature_matrix(frac_coords, atomic_numbers)
    one_hot_part = out[0, 3:103]
    assert one_hot_part[5].item() == 1.0
    assert one_hot_part.sum().item() == 1.0


def test_build_atom_feature_matrix_frac_coords():
    N = 2
    frac_coords = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    atomic_numbers = torch.tensor([1, 2], dtype=torch.long)
    out = build_encoder1_atom_feature_matrix(frac_coords, atomic_numbers)
    assert torch.allclose(out[:, :3], frac_coords)


def test_legacy_build_atom_feature_matrix_aliases_encoder1_features():
    frac_coords = torch.rand(2, 3)
    atomic_numbers = torch.tensor([1, 8], dtype=torch.long)
    out = build_atom_feature_matrix(frac_coords, atomic_numbers)
    assert out.shape == (2, 103)


def _make_test_structure():
    lattice = Lattice.cubic(4.0)
    species = ["Na", "Cl"]
    coords = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    return Structure(lattice, species, coords)


def test_structure_to_tensors_shapes():
    structure = _make_test_structure()
    frac_coords, atomic_numbers, family_one_hot, num_atoms = structure_to_tensors(
        structure
    )
    assert frac_coords.shape == (2, 3)
    assert atomic_numbers.shape == (2,)
    assert family_one_hot.shape == (7,)
    assert num_atoms == 2


def test_structure_to_tensors_frac_coords_range():
    structure = _make_test_structure()
    frac_coords, _, _, _ = structure_to_tensors(structure)
    assert (frac_coords >= 0.0).all() and (frac_coords <= 1.0).all()


def test_structure_to_tensors_atomic_numbers():
    structure = _make_test_structure()
    _, atomic_numbers, _, _ = structure_to_tensors(structure)
    # Na=11, Cl=17
    nums = set(atomic_numbers.tolist())
    assert nums == {11, 17}


def test_structure_to_tensors_family_one_hot_shape():
    structure = _make_test_structure()
    _, _, family_one_hot, _ = structure_to_tensors(structure)
    assert family_one_hot.shape == (7,)
    assert family_one_hot.dtype == torch.float32
    assert family_one_hot.sum().item() == 1.0


def test_crystal_family_one_hot_all_systems():
    cases = [
        (Lattice.from_parameters(4, 5, 6, 80, 75, 70), 0),
        (Lattice.monoclinic(4, 5, 6, 110), 1),
        (Lattice.orthorhombic(4, 5, 6), 2),
        (Lattice.tetragonal(4, 6), 3),
        (Lattice.rhombohedral(4, 75), 4),
        (Lattice.hexagonal(4, 6), 5),
        (Lattice.cubic(4), 6),
    ]
    for lattice, expected_index in cases:
        one_hot = crystal_family_one_hot(lattice)
        assert one_hot.shape == (7,)
        assert one_hot.argmax().item() == expected_index
        assert one_hot.sum().item() == 1.0


def test_spatial_block_visible_mask_removes_atoms_inside_toric_box():
    frac_coords = torch.tensor(
        [
            [0.95, 0.50, 0.50],
            [0.05, 0.50, 0.50],
            [0.50, 0.50, 0.50],
        ],
        dtype=torch.float32,
    )
    center = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float32)
    visible = spatial_block_visible_mask(frac_coords, center, box_size=0.2)
    assert visible.tolist() == [False, False, True]


def test_sample_spatial_block_keeps_at_least_one_atom_visible():
    frac_coords = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32)
    visible, mask_box = sample_spatial_block(
        frac_coords,
        center=torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32),
        box_size=1.0,
    )
    assert visible.tolist() == [True]
    assert mask_box.shape == (4,)


def test_sample_spatial_block_masks_one_atom_when_possible():
    frac_coords = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.9, 0.9, 0.9],
        ],
        dtype=torch.float32,
    )
    visible, mask_box = sample_spatial_block(
        frac_coords,
        center=torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32),
        box_size=0.01,
    )

    assert visible.shape == (2,)
    assert mask_box.shape == (4,)
    assert visible.any()
    assert (~visible).any()


def test_collate_crystals_returns_full_and_masked_encoder_inputs():
    samples = []
    for material_id, lattice in [
        ("a", Lattice.cubic(4)),
        ("b", Lattice.hexagonal(4, 6)),
    ]:
        structure = Structure(
            lattice,
            ["Na", "Cl", "O"],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        )
        frac_coords, atomic_numbers, family_one_hot, num_atoms = structure_to_tensors(
            structure
        )
        samples.append(
            {
                "material_id": material_id,
                "frac_coords": frac_coords,
                "atomic_numbers": atomic_numbers,
                "family_one_hot": family_one_hot,
                "encoder2_atom_features": build_encoder2_atom_feature_matrix(
                    frac_coords, atomic_numbers, family_one_hot
                ),
                "ef_per_atom": torch.tensor(-1.0, dtype=torch.float32),
                "num_atoms": torch.tensor(num_atoms, dtype=torch.long),
            }
        )

    batch = collate_crystals(samples)

    assert batch["encoder1_atom_features"].shape[0] == 2
    assert batch["encoder1_atom_features"].shape[-1] == 103
    assert batch["encoder2_atom_features"].shape == (2, 3, 110)
    assert batch["encoder1_atom_mask"].shape[0] == 2
    assert batch["encoder2_atom_mask"].tolist() == [
        [True, True, True],
        [True, True, True],
    ]
    assert batch["family_one_hot"].shape == (2, 7)
    assert batch["mask_box"].shape == (2, 4)


def test_build_energy_weight_matrix_shape():
    ef = torch.tensor([-1.0, -2.0, -0.5])
    out = build_energy_weight_matrix(ef)
    assert out.shape == (3, 3)


def test_build_energy_weight_matrix_diagonal_ones():
    ef = torch.tensor([-1.0, -2.0, -0.5])
    out = build_energy_weight_matrix(ef)
    assert torch.allclose(out.diagonal(), torch.ones(3))


def test_build_energy_weight_matrix_off_diagonal_range():
    ef = torch.tensor([-1.0, -2.0, -0.5])
    out = build_energy_weight_matrix(ef)
    diag_mask = ~torch.eye(3, dtype=torch.bool)
    off_diag = out[diag_mask]
    assert (off_diag >= 0.0).all() and (off_diag < 1.0).all()


def test_build_energy_weight_matrix_symmetry():
    ef = torch.tensor([-1.0, -2.0, -0.5])
    out = build_energy_weight_matrix(ef)
    assert torch.allclose(out, out.T)


def test_build_energy_weight_matrix_values():
    ef = torch.tensor([0.0, 1.0])
    out = build_energy_weight_matrix(ef)
    expected_off = 1.0 - torch.exp(torch.tensor(-1.0))
    assert torch.allclose(out[0, 1], expected_off, atol=1e-5)
    assert torch.allclose(out[1, 0], expected_off, atol=1e-5)


def test_superconductor_dataset_deduplicate_data_keeps_first_material_id():
    data = pd.DataFrame(
        {
            "material_id": ["mp-1", "mp-1", "mp-2"],
            "cif": ["first", "duplicate", "other"],
            "ef_per_atom": [-1.0, -1.0, -2.0],
        }
    )

    deduplicated = SuperconductorDataset._deduplicate_data(data)

    assert list(deduplicated["material_id"]) == ["mp-1", "mp-2"]
    assert list(deduplicated["cif"]) == ["first", "other"]


def test_create_dataloader_can_drop_last_partial_batch(tmp_path):
    structure = _make_test_structure()
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "material_id": ["mp-1", "mp-2", "mp-3"],
            "cif": [structure.to(fmt="cif")] * 3,
            "ef_per_atom": [-1.0, -2.0, -3.0],
        }
    ).to_csv(csv_path, index=False)

    dataloader = create_dataloader(
        csv_path,
        batch_size=2,
        shuffle=False,
        drop_last=True,
    )

    batches = list(dataloader)
    assert len(batches) == 1
    assert batches[0]["encoder1_atom_features"].shape[0] == 2


def test_public_api_all_functions_importable():
    from crystal_matrices import (
        build_atom_feature_matrix,
        build_encoder1_atom_feature_matrix,
        build_encoder2_atom_feature_matrix,
        build_energy_weight_matrix,
        compute_lattice_6d,
        crystal_family_one_hot,
        sample_spatial_block,
        spatial_block_visible_mask,
        structure_to_tensors,
    )
    assert callable(compute_lattice_6d)
    assert callable(build_atom_feature_matrix)
    assert callable(build_encoder1_atom_feature_matrix)
    assert callable(build_encoder2_atom_feature_matrix)
    assert callable(structure_to_tensors)
    assert callable(build_energy_weight_matrix)
    assert callable(crystal_family_one_hot)
    assert callable(spatial_block_visible_mask)
    assert callable(sample_spatial_block)


def test_end_to_end_structure_to_atom_matrix():
    """Full pipeline: Structure → encoder matrices."""
    from crystal_matrices import (
        build_encoder1_atom_feature_matrix,
        build_encoder2_atom_feature_matrix,
        structure_to_tensors,
    )

    lattice = Lattice.cubic(4.0)
    structure = Structure(
        lattice,
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    frac_coords, atomic_numbers, family_one_hot, num_atoms = structure_to_tensors(
        structure
    )
    encoder1 = build_encoder1_atom_feature_matrix(frac_coords, atomic_numbers)
    encoder2 = build_encoder2_atom_feature_matrix(
        frac_coords, atomic_numbers, family_one_hot
    )

    assert encoder1.shape == (num_atoms, 103)
    assert encoder2.shape == (num_atoms, 110)
    assert encoder1.dtype == torch.float32
    assert encoder2.dtype == torch.float32
    # One-hot part sums to 1 per atom
    assert torch.allclose(encoder1[:, 3:103].sum(dim=1), torch.ones(num_atoms))
    assert torch.allclose(encoder2[:, 103:], family_one_hot.expand(num_atoms, -1))
