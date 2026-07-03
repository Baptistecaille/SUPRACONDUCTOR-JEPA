import torch

from models.property_jepa.data.property_schema import NUM_PROPERTY_TYPES, property_type_id
from models.property_jepa.model.property_encoder import (
    PropertyEncoder,
    TargetPropertyEncoder,
    masked_mean_pool,
)


def test_property_encoder_outputs_hidden_dim_embedding():
    encoder = PropertyEncoder(hidden_dim=16, num_property_types=NUM_PROPERTY_TYPES)
    type_id = torch.tensor([0, 1, 2])
    value = torch.tensor([1.0, -0.5, 0.2])

    out = encoder(type_id, value)

    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()


def test_property_encoder_supports_multi_dim_batches():
    encoder = PropertyEncoder(hidden_dim=8, num_property_types=NUM_PROPERTY_TYPES)
    type_id = torch.randint(0, NUM_PROPERTY_TYPES, (2, 5))
    value = torch.randn(2, 5)

    out = encoder(type_id, value)

    assert out.shape == (2, 5, 8)


def test_property_encoder_rejects_out_of_range_type_id():
    encoder = PropertyEncoder(hidden_dim=8, num_property_types=NUM_PROPERTY_TYPES)
    type_id = torch.tensor([NUM_PROPERTY_TYPES])
    value = torch.tensor([0.0])

    try:
        encoder(type_id, value)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_target_property_encoder_has_frozen_parameters():
    target_encoder = TargetPropertyEncoder(hidden_dim=8, num_property_types=NUM_PROPERTY_TYPES)

    assert all(not p.requires_grad for p in target_encoder.parameters())


def test_property_type_id_roundtrip():
    idx = property_type_id("tc_experimental")
    assert isinstance(idx, int)
    assert 0 <= idx < NUM_PROPERTY_TYPES


def test_property_type_id_rejects_unknown_name():
    try:
        property_type_id("not_a_real_property")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_masked_mean_pool_ignores_masked_slots():
    embeddings = torch.tensor(
        [
            [[1.0, 1.0], [100.0, 100.0], [3.0, 3.0]],
        ]
    )
    mask = torch.tensor([[True, False, True]])

    pooled = masked_mean_pool(embeddings, mask)

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_masked_mean_pool_returns_zero_for_all_masked_row():
    embeddings = torch.randn(1, 4, 3)
    mask = torch.zeros(1, 4, dtype=torch.bool)

    pooled = masked_mean_pool(embeddings, mask)

    assert torch.allclose(pooled, torch.zeros(1, 3))
