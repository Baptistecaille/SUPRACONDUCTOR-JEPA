import torch

from models.foundation_jepa.model.backbone import MLP, FoundationTransformer, MaskedMHA, PreNormEncoderLayer


def test_mlp_shapes_and_layers():
    mlp = MLP(10, 32, 4, n_layers=3)
    x = torch.randn(5, 10)
    out = mlp(x)
    assert out.shape == (5, 4)


def test_mlp_rejects_too_few_layers():
    try:
        MLP(10, 32, 4, n_layers=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_masked_mha_output_shape():
    mha = MaskedMHA(attn_head=4, dim=16, dropout=0.0)
    b, n = 3, 6
    q = k = v = torch.randn(b, n, 16)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[0, 4:] = False
    out = mha(q, k, v, mask)
    assert out.shape == (b, n, 16)


def test_prenorm_encoder_layer_zeroes_padding():
    layer = PreNormEncoderLayer(dim=16, attn_head=4, dropout=0.0)
    b, n = 2, 5
    h = torch.randn(b, n, 16)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[:, 3:] = False
    out = layer(h, mask)
    assert torch.allclose(out[~mask], torch.zeros_like(out[~mask]))


def test_foundation_transformer_forward_shape_and_masking():
    torch.manual_seed(0)
    model = FoundationTransformer(hidden_dim=16, layers=2, attn_head=4, dropout=0.0, max_len=20)
    b, n = 3, 7
    h = torch.randn(b, n, 16)
    mask = torch.ones(b, n, dtype=torch.bool)
    mask[0, 5:] = False
    mask[1, 6:] = False
    out = model(h, mask)
    assert out.shape == (b, n + 1, 16)
    # CLS token (position 0) always visible/non-zero for every sample
    assert out[:, 0].abs().sum(dim=-1).min() > 0


def test_foundation_transformer_rejects_overlong_sequence():
    model = FoundationTransformer(hidden_dim=8, layers=1, attn_head=2, dropout=0.0, max_len=5)
    h = torch.randn(1, 10, 8)
    mask = torch.ones(1, 10, dtype=torch.bool)
    try:
        model(h, mask)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_foundation_transformer_padding_invariance():
    """Two samples with identical real atoms but different amounts of padding
    should give the same CLS embedding (padding must not leak into attention)."""
    torch.manual_seed(1)
    model = FoundationTransformer(hidden_dim=16, layers=2, attn_head=4, dropout=0.0, max_len=20)
    model.eval()

    n_real = 4
    h_real = torch.randn(1, n_real, 16)

    h_a = torch.cat([h_real, torch.zeros(1, 2, 16)], dim=1)
    mask_a = torch.tensor([[True] * n_real + [False] * 2])

    h_b = torch.cat([h_real, torch.randn(1, 5, 16)], dim=1)
    mask_b = torch.tensor([[True] * n_real + [False] * 5])

    with torch.no_grad():
        out_a = model(h_a, mask_a)[:, 0]
        out_b = model(h_b, mask_b)[:, 0]

    assert torch.allclose(out_a, out_b, atol=1e-4)
