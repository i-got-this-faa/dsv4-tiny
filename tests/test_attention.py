"""Tests for DSV4Attention."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.attention import DSV4Attention


def test_swa_attention_forward():
    cfg = DSV4TinyConfig()
    attn = DSV4Attention(cfg, layer_idx=0)
    assert attn.layer_type == "swa"
    b, s, d = 2, 8, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x)
    assert out.shape == (b, s, d)


def test_csa_attention_forward():
    cfg = DSV4TinyConfig()
    attn = DSV4Attention(cfg, layer_idx=2)
    assert attn.layer_type == "csa"
    b, s, d = 1, 4, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x)
    assert out.shape == (b, s, d)


def test_hca_attention_forward():
    cfg = DSV4TinyConfig()
    attn = DSV4Attention(cfg, layer_idx=3)
    assert attn.layer_type == "hca"
    b, s, d = 1, 4, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x)
    assert out.shape == (b, s, d)


def test_attention_with_cache():
    cfg = DSV4TinyConfig()
    from dsv4_tiny.cache import DSV4Cache
    cache = DSV4Cache(cfg)
    attn = DSV4Attention(cfg, layer_idx=0)
    b, s, d = 1, 32, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x, cache=cache)
    assert out.shape == (b, s, d)


def test_attention_decomposed_query():
    cfg = DSV4TinyConfig()
    attn = DSV4Attention(cfg, layer_idx=2)  # CSA
    b, s, d = 1, 4, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = attn(x)
    assert out.shape == (b, s, d)
    # CSA has W_DQ and W_UQ
    assert hasattr(attn, "W_DQ")
    assert hasattr(attn, "W_UQ")


def test_attention_sink_token():
    cfg = DSV4TinyConfig()
    attn = DSV4Attention(cfg, layer_idx=0)
    assert attn.sink_token.shape == (1, 1, cfg.head_dim)


def test_o_gate_present():
    cfg = DSV4TinyConfig()
    assert cfg.attn_output_gate
    attn = DSV4Attention(cfg, layer_idx=0)
    assert hasattr(attn, "o_gate")
