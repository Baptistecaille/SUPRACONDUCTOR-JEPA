import torch

from models.property_jepa.model.predictor import PropertyPredictor


def test_predictor_outputs_hidden_dim_embedding():
    predictor = PropertyPredictor(hidden_dim=16, num_property_types=14)
    z_crystal = torch.randn(4, 16)
    known_context = torch.randn(4, 16)
    query_type = torch.randint(0, 14, (4,))

    out = predictor(z_crystal, known_context, query_type)

    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_predictor_rejects_hidden_dim_mismatch():
    predictor = PropertyPredictor(hidden_dim=16, num_property_types=14)
    z_crystal = torch.randn(4, 8)
    known_context = torch.randn(4, 16)
    query_type = torch.randint(0, 14, (4,))

    try:
        predictor(z_crystal, known_context, query_type)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_predictor_rejects_known_context_shape_mismatch():
    predictor = PropertyPredictor(hidden_dim=16, num_property_types=14)
    z_crystal = torch.randn(4, 16)
    known_context = torch.randn(3, 16)
    query_type = torch.randint(0, 14, (4,))

    try:
        predictor(z_crystal, known_context, query_type)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_predictor_is_sensitive_to_query_type():
    """Different query property types must (generically) yield different predictions,
    confirming the predictor actually conditions on `query_property_type_id`
    rather than ignoring it."""
    torch.manual_seed(0)
    predictor = PropertyPredictor(hidden_dim=16, num_property_types=14)
    z_crystal = torch.randn(1, 16)
    known_context = torch.randn(1, 16)

    out_a = predictor(z_crystal, known_context, torch.tensor([0]))
    out_b = predictor(z_crystal, known_context, torch.tensor([1]))

    assert not torch.allclose(out_a, out_b)
