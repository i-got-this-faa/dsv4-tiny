"""Tests for training objectives and reconstruction loss."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.cache import DSV4Cache


def test_reconstruction_loss_direct():
    """Reconstruction loss is non-zero when compression happens."""
    cfg = DSV4TinyConfig()
    cache = DSV4Cache(cfg)

    # Push exactly block_size tokens to trigger compression
    kv = torch.randn(cfg.num_key_value_heads, cfg.head_dim)
    hs = torch.randn(cfg.hidden_size)
    for i in range(cfg.block_alignment):
        cache.push(0, kv, kv, hidden_state=hs)

    recon_loss = cache.reconstruction_loss
    assert recon_loss is not None, "recon_loss should be computed after compression"
    assert recon_loss.item() > 0, "recon_loss should be > 0"
