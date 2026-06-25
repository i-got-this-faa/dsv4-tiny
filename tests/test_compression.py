"""Tests for CSACompressor and HCACompressor."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.compression import CSACompressor, HCACompressor


def test_csa_compressor_output_shape():
    cfg = DSV4TinyConfig()
    compressor = CSACompressor(cfg)
    b, m, d = 2, cfg.csa_block_size, cfg.hidden_size
    hidden = torch.randn(b, m, d)
    block_mask = torch.ones(b, m, dtype=torch.bool)

    c_comp = compressor(hidden, block_mask)
    expected_dim = cfg.csa_compressed_dim * cfg.csa_groups + cfg.csa_intermediate
    assert c_comp.shape == (b, expected_dim), f"Expected ({b}, {expected_dim}), got {c_comp.shape}"


def test_csa_first_block_boundary():
    cfg = DSV4TinyConfig()
    compressor = CSACompressor(cfg)
    b, m, d = 1, cfg.csa_block_size, cfg.hidden_size
    hidden = torch.randn(b, m, d)
    block_mask = torch.ones(b, m, dtype=torch.bool)

    # First block: should use only Ca stream
    c_first = compressor(hidden, block_mask, first_block=True)
    c_normal = compressor(hidden, block_mask, first_block=False)

    # Both should succeed with the same shape
    expected_dim = cfg.csa_compressed_dim * cfg.csa_groups + cfg.csa_intermediate
    assert c_first.shape == (b, expected_dim)
    assert c_normal.shape == (b, expected_dim)


def test_csa_block_mask():
    cfg = DSV4TinyConfig()
    compressor = CSACompressor(cfg)
    b, m, d = 1, cfg.csa_block_size, cfg.hidden_size
    hidden = torch.randn(b, m, d)

    # Partial block (only 3 of 4 tokens valid)
    block_mask = torch.tensor([[True, True, True, False]])
    c_comp = compressor(hidden, block_mask)
    expected_dim = cfg.csa_compressed_dim * cfg.csa_groups + cfg.csa_intermediate
    assert c_comp.shape == (b, expected_dim)


def test_hca_compressor_output_shape():
    cfg = DSV4TinyConfig()
    compressor = HCACompressor(cfg)
    b, m, d = 2, cfg.hca_block_size, cfg.hidden_size
    hidden = torch.randn(b, m, d)
    block_mask = torch.ones(b, m, dtype=torch.bool)

    c_comp = compressor(hidden, block_mask)
    expected_dim = cfg.hca_compressed_dim
    assert c_comp.shape == (b, expected_dim), f"Expected ({b}, {expected_dim}), got {c_comp.shape}"


def test_compressor_gradient_flow():
    cfg = DSV4TinyConfig()
    compressor = CSACompressor(cfg)
    b, m, d = 1, cfg.csa_block_size, cfg.hidden_size
    hidden = torch.randn(b, m, d, requires_grad=True)
    block_mask = torch.ones(b, m, dtype=torch.bool)

    c_comp = compressor(hidden, block_mask)
    loss = c_comp.sum()
    loss.backward()

    assert compressor.W_aKV.weight.grad is not None
    assert compressor.W_aZ.weight.grad is not None
    assert compressor.Ba.grad is not None
