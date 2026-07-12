"""Tests for DSV4TinyForCausalLM (hierarchical memory mode)."""

import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.model import DSV4TinyForCausalLM, DecoderLayer, RMSNorm


def test_model_creation():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)
    assert model is not None
    assert len(model.layers) == 24


def test_model_forward_pass():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(torch.bfloat16)

    input_ids = torch.randint(0, 1000, (1, 16))
    outputs = model(input_ids, use_cache=True)
    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 16, cfg.vocab_size)


def test_model_loss():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(torch.bfloat16)

    input_ids = torch.randint(0, 1000, (1, 16))
    labels = input_ids.clone()
    outputs = model(input_ids, labels=labels, use_cache=True)
    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"]).item()


def test_model_without_cache():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(torch.bfloat16)

    input_ids = torch.randint(0, 1000, (1, 16))
    outputs = model(input_ids, use_cache=False)
    assert outputs["logits"].shape == (1, 16, cfg.vocab_size)


def test_model_gradient_flow():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)

    # Enable gradients
    model.train()
    input_ids = torch.randint(0, 1000, (1, 8))
    outputs = model(input_ids, labels=input_ids.clone(), use_cache=False)
    loss = outputs["loss"]
    loss.backward()

    # Check that at least some parameters got gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "No parameters received gradients"


def test_decoder_layer():
    cfg = DSV4TinyConfig()
    layer = DecoderLayer(cfg, layer_idx=0)
    b, s, d = 2, 8, cfg.hidden_size
    x = torch.randn(b, s, d)
    out = layer(x)
    assert out.shape == (b, s, d)


def test_rms_norm():
    norm = RMSNorm(1024, eps=1e-6)
    x = torch.randn(2, 8, 1024)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_model_generate():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(torch.bfloat16)

    input_ids = torch.randint(0, 1000, (1, 8))
    output_ids = model.generate(input_ids, max_new_tokens=5, temperature=0)
    assert output_ids.shape[1] == 8 + 5, f"Expected 13 tokens, got {output_ids.shape[1]}"


def test_all_layers_have_all_projections():
    """Every layer should have all projections in hierarchical mode."""
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)

    from dsv4_tiny.attention import DSV4Attention

    for i, layer in enumerate(model.layers):
        assert isinstance(layer.self_attn, DSV4Attention)
        attn = layer.self_attn
        # All layers have both SWA and compressed projections
        assert hasattr(attn, "k_proj"), f"Layer {i} missing k_proj"
        assert hasattr(attn, "v_proj"), f"Layer {i} missing v_proj"
        assert hasattr(attn, "kv_decompress"), f"Layer {i} missing kv_decompress"
        assert hasattr(attn, "W_UQ"), f"Layer {i} missing W_UQ"
        # All layers have three sink tokens
        assert hasattr(attn, "sink_swa"), f"Layer {i} missing sink_swa"
        assert hasattr(attn, "sink_csa"), f"Layer {i} missing sink_csa"
        assert hasattr(attn, "sink_hca"), f"Layer {i} missing sink_hca"
        # All layers have combination weights
        assert hasattr(attn, "alpha_swa"), f"Layer {i} missing alpha_swa"
        assert hasattr(attn, "alpha_csa"), f"Layer {i} missing alpha_csa"
        assert hasattr(attn, "alpha_hca"), f"Layer {i} missing alpha_hca"


def test_cache_stats_in_output():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(torch.bfloat16)

    input_ids = torch.randint(0, 1000, (1, 16))
    outputs = model(input_ids, use_cache=True)
    assert "cache_stats" in outputs
    stats = outputs["cache_stats"]
    assert "csa_block_counts" in stats
    assert "hca_block_counts" in stats
