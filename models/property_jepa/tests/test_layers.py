import torch

from models.property_jepa.model.layers import MLP


def test_mlp_output_shape():
    mlp = MLP(input_dim=4, hidden_dim=8, output_dim=2, layers=3)
    x = torch.randn(5, 4)

    out = mlp(x)

    assert out.shape == (5, 2)


def test_mlp_rejects_fewer_than_two_layers():
    try:
        MLP(input_dim=4, hidden_dim=8, output_dim=2, layers=1)
        assert False, "expected ValueError"
    except ValueError:
        pass
