"""Tests for DSV4TinyForCausalLM."""

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
    outputs = model(input_ids, labels=input_ids.clone(), use_cache=True)

    assert "loss" in outputs
    assert torch.isfinite(outputs["loss"]).item()


def test_model_without_cache():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(torch.bfloat16)

    input_ids = torch.randint(0, 1000, (1, 16))
    outputs = model(input_ids, use_cache=False)

    assert "logits" in outputs
    assert outputs["logits"].shape == (1, 16, cfg.vocab_size)


def test_model_gradient_flow():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(torch.bfloat16)

    input_ids = torch.randint(0, 1000, (1, 8))
    outputs = model(input_ids, labels=input_ids.clone(), use_cache=True)

    loss = outputs["loss"]
    loss.backward()

    # Check that some parameters got gradients
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No parameters received gradients"


def test_decoder_layer():
    cfg = DSV4TinyConfig()
    layer = DecoderLayer(cfg, layer_idx=0)
    b, s, d = 1, 8, cfg.hidden_size
    x = torch.randn(b, s, d)

    out = layer(x)
    assert out.shape == (b, s, d)


def test_rms_norm():
    norm = RMSNorm(1024, eps=1e-6)
    x = torch.randn(1, 8, 1024)
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


def test_attention_layer_types():
    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)

    from dsv4_tiny.attention import DSV4Attention

    for i, layer in enumerate(model.layers):
        assert isinstance(layer.self_attn, DSV4Attention)
        layer_type = layer.self_attn.layer_type
        assert layer_type in ("swa", "csa", "hca"), f"Layer {i}: {layer_type}"
        if layer_type == "swa":
            assert hasattr(layer.self_attn, "k_proj"), f"SWA layer {i} missing k_proj"
        else:
            assert hasattr(layer.self_attn, "kv_decompress"), f"CSA/HCA layer {i} missing kv_decompress"
