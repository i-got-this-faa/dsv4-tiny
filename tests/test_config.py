"""Tests for DSV4TinyConfig."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dsv4_tiny.config import DSV4TinyConfig


def test_default_config():
    cfg = DSV4TinyConfig()
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 8
    assert cfg.num_key_value_heads == 2
    assert cfg.head_dim == 256
    assert len(cfg.swa_layers) == 2
    assert len(cfg.csa_layers) == 11
    assert len(cfg.hca_layers) == 11
    assert cfg.csa_block_size == 4
    assert cfg.hca_block_size == 128
    assert cfg.block_alignment == 128


def test_layer_assignment():
    cfg = DSV4TinyConfig()
    for i in range(24):
        t = cfg.layer_type(i)
        assert t in ("swa", "csa", "hca"), f"Layer {i}: invalid type {t}"
    assert cfg.layer_type(0) == "swa"
    assert cfg.layer_type(1) == "swa"
    assert cfg.layer_type(2) == "csa"
    assert cfg.layer_type(3) == "hca"
    assert cfg.layer_type(4) == "csa"


def test_validates_layer_count():
    passed = False
    try:
        # Creating with wrong layer count should fail
        DSV4TinyConfig(
            swa_layers=(0,),
            csa_layers=(1, 2),
            hca_layers=(3,),
            num_hidden_layers=24,
        )
    except AssertionError:
        passed = True
    assert passed, "Should have raised AssertionError for wrong layer count"


def test_rope_dim():
    cfg = DSV4TinyConfig()
    assert cfg.rope_dim == 64
    assert cfg.partial_rotary_factor == 0.25
    assert int(cfg.head_dim * cfg.partial_rotary_factor) == cfg.rope_dim


def test_to_dict():
    cfg = DSV4TinyConfig()
    d = cfg.to_dict()
    assert d["hidden_size"] == 1024
    assert d["num_hidden_layers"] == 24
