"""Tests for DSV4Attention (hierarchical memory mode)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.attention import DSV4Attention


def _make_attn(cfg, layer_idx=0):
    """Helper to create an attention module."""
    return DSV4Attention(cfg, layer_idx)


def test_swa_attention_forward():
    cfg = DSV4TinyConfig()
    attn = _make_attn(cfg, layer_idx=0)
    assert hasattr(attn, "k_proj"), "All layers now have k_proj"
    b, s, d = 2, 8, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x)
    assert out.shape == (b, s, d)


def test_csa_attention_forward():
    cfg = DSV4TinyConfig()
    attn = _make_attn(cfg, layer_idx=2)
    b, s, d = 1, 4, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x)
    assert out.shape == (b, s, d)


def test_hca_attention_forward():
    cfg = DSV4TinyConfig()
    attn = _make_attn(cfg, layer_idx=3)
    b, s, d = 1, 4, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x)
    assert out.shape == (b, s, d)


def test_hca_attention_forward():
    cfg = DSV4TinyConfig()
    attn = _make_attn(cfg, layer_idx=3)
    b, s, d = 1, 4, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x)
    assert out.shape == (b, s, d)


def test_attention_with_cache():
    cfg = DSV4TinyConfig()
    from dsv4_tiny.cache import DSV4Cache
    cache = DSV4Cache(cfg)
    attn = _make_attn(cfg, layer_idx=0)
    b, s, d = 1, 32, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x, cache=cache)
    assert out.shape == (b, s, d)


def test_attention_decomposed_query():
    cfg = DSV4TinyConfig()
    attn = _make_attn(cfg, layer_idx=2)
    b, s, d = 1, 4, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x)
    assert out.shape == (b, s, d)
    # All layers now have W_DQ and W_UQ
    assert hasattr(attn, "W_DQ")
    assert hasattr(attn, "W_UQ")


def test_attention_sink_tokens():
    cfg = DSV4TinyConfig()
    attn = _make_attn(cfg, layer_idx=0)
    head_dim = cfg.head_dim
    assert attn.sink_swa.shape == (1, 1, head_dim)
    assert attn.sink_csa.shape == (1, 1, head_dim)
    assert attn.sink_hca.shape == (1, 1, head_dim)


def test_attention_combination_weights():
    cfg = DSV4TinyConfig()
    attn = _make_attn(cfg, layer_idx=0)
    assert hasattr(attn, "alpha_swa")
    assert hasattr(attn, "alpha_csa")
    assert hasattr(attn, "alpha_hca")
def test_all_projections_present():
    cfg = DSV4TinyConfig()
    for layer_idx in range(3):
        attn = _make_attn(cfg, layer_idx)
        assert hasattr(attn, "k_proj"), f"Layer {layer_idx} missing k_proj"
        assert hasattr(attn, "v_proj"), f"Layer {layer_idx} missing v_proj"
        assert hasattr(attn, "W_UQ"), f"Layer {layer_idx} missing W_UQ"


def test_o_gate_present():
    cfg = DSV4TinyConfig()
    assert cfg.attn_output_gate
    attn = _make_attn(cfg, layer_idx=0)
    assert hasattr(attn, "o_gate")
