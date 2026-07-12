"""Tests for DSV4TinyConfig (hierarchical memory mode)."""

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
    assert cfg.use_hierarchical is True
    assert cfg.csa_block_size == 4
    assert cfg.hca_block_size == 128
    assert cfg.block_alignment == 128


def test_sink_flags():
    cfg = DSV4TinyConfig()
    assert cfg.sink_swa is True
    assert cfg.sink_csa is True
    assert cfg.sink_hca is True


def test_latent_hyperparams():
    cfg = DSV4TinyConfig()
    assert cfg.latent_memory_size == 128
    assert cfg.latent_slots_per_chunk == 4
    assert cfg.encoder_hidden == 256
    assert cfg.decoder_hidden == 256


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
    assert d["use_hierarchical"] is True
