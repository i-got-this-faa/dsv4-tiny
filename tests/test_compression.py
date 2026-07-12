"""Tests for LatentMemoryEncoder and LatentMemoryDecoder."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.compression import LatentMemoryEncoder, LatentMemoryDecoder


def test_latent_encoder_output_shape():
    cfg = DSV4TinyConfig()
    b, chunk_size, d = 2, cfg.block_alignment, cfg.hidden_size
    encoder = LatentMemoryEncoder(
        hidden_size=d,
        num_slots=cfg.latent_slots_per_chunk,
        latent_size=cfg.latent_memory_size,
        encoder_hidden=cfg.encoder_hidden,
        chunk_size=chunk_size,
    )
    hidden = torch.randn(b, chunk_size, d)
    latents = encoder(hidden)
    assert latents.shape == (b, cfg.latent_slots_per_chunk, cfg.latent_memory_size), \
        f"Expected ({b}, {cfg.latent_slots_per_chunk}, {cfg.latent_memory_size}), got {latents.shape}"


def test_latent_decoder_output_shape():
    cfg = DSV4TinyConfig()
    decoder = LatentMemoryDecoder(
        latent_size=cfg.latent_memory_size,
        head_dim=cfg.head_dim,
    )
    b = 2
    latent = torch.randn(b, cfg.latent_memory_size)
    k, v, confidence, recon = decoder(latent)
    assert k.shape == (b, cfg.head_dim), f"Expected ({b}, {cfg.head_dim}), got {k.shape}"
    assert v.shape == (b, cfg.head_dim)
    assert confidence.shape == (b, 1)
    assert recon.shape == (b, cfg.block_alignment * cfg.hidden_size), \
        f"Expected ({b}, {cfg.block_alignment * cfg.hidden_size}), got {recon.shape}"


def test_latent_gradient_flow():
    cfg = DSV4TinyConfig()
    b, chunk_size, d = 1, cfg.block_alignment, cfg.hidden_size
    encoder = LatentMemoryEncoder(
        hidden_size=d,
        num_slots=cfg.latent_slots_per_chunk,
        latent_size=cfg.latent_memory_size,
        encoder_hidden=cfg.encoder_hidden,
        chunk_size=chunk_size,
    )
    decoder = LatentMemoryDecoder(
        latent_size=cfg.latent_memory_size,
        head_dim=cfg.head_dim,
    )
    hidden = torch.randn(b, chunk_size, d, requires_grad=True)
    latents = encoder(hidden)
    latent = latents.mean(dim=1)  # pool slots
    k, v, confidence, recon = decoder(latent)
    loss = (k.mean() + v.mean() + recon.mean())
    loss.backward()

    assert encoder.token_proj.weight.grad is not None, "Encoder token_proj should get gradients"
    assert decoder.proj.weight.grad is not None, "Decoder proj should get gradients"
    assert decoder.reconstruct.weight.grad is not None, "Decoder reconstruct should get gradients"

def test_encoder_confidence_range():
    """Confidence should be in (0, 1) after sigmoid."""
    cfg = DSV4TinyConfig()
    decoder = LatentMemoryDecoder(
        latent_size=cfg.latent_memory_size,
        head_dim=cfg.head_dim,
    )
    b = 4
    latent = torch.randn(b, cfg.latent_memory_size) * 10  # large values to test sigmoid
    _, _, confidence, _ = decoder(latent)
    assert confidence.min() >= 0.0, "Confidence should be >= 0"
    assert confidence.max() <= 1.0, "Confidence should be <= 1"
