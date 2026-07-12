"""Compression modules for DSV4-Tiny: LatentMemoryEncoder and LatentMemoryDecoder.

Learned latent-vector compression replaces the old dual-stream attention pooling
(CSACompressor) and single-stream compression (HCACompressor).

Architecture:
    hidden (chunk_size x d) -> Encoder -> multi-slot latents -> Decoder -> (K, V, confidence)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentMemoryEncoder(nn.Module):
    """Encodes a chunk of hidden states into multi-slot latent memories.

    Efficient architecture: projects each token independently, aggregates,
    then projects to slots. Avoids the massive flatten → one-shot MLP.

    Args:
        hidden_size: d (1024)
        num_slots: latent_slots_per_chunk (4)
        latent_size: latent_memory_size (128)
        encoder_hidden: intermediate dim (256)
        chunk_size: tokens per chunk (128, unused in per-token design)
    """

    def __init__(self, hidden_size, num_slots, latent_size, encoder_hidden, chunk_size=None):
        super().__init__()
        self.num_slots = num_slots
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        # Per-token down-projection: d → encoder_hidden
        self.token_proj = nn.Linear(hidden_size, encoder_hidden, bias=False)
        self.token_norm = nn.LayerNorm(encoder_hidden)
        # Aggregate → slots: encoder_hidden → num_slots * latent_size
        self.slot_proj = nn.Linear(encoder_hidden, num_slots * latent_size)

    def forward(self, hidden_chunk):
        # hidden_chunk: (b, chunk_size, d)
        b, m, d = hidden_chunk.shape
        # Per-token: (b, m, d) → (b, m, encoder_hidden)
        x = F.silu(self.token_proj(hidden_chunk))
        x = self.token_norm(x)
        # Mean-pool over tokens: (b, encoder_hidden)
        x = x.mean(dim=1)
        # Project to slots: (b, num_slots * latent_size)
        latents = self.slot_proj(x)
        latents = latents.view(b, self.num_slots, self.latent_size)
        return latents


class LatentMemoryDecoder(nn.Module):
    """Decodes a latent memory vector into K, V, confidence score, and reconstructed hidden.

    Architecture:
        latent (latent_size)
          -> LayerNorm
          -> Linear(head_dim * 2 + 1)    # K + V + confidence
          -> Linear(chunk_size * hidden_size)  # reconstruction (for training loss)
    """

    def __init__(self, latent_size, head_dim, chunk_size=128, hidden_size=1024):
        super().__init__()
        self.latent_size = latent_size
        self.head_dim = head_dim
        self.norm = nn.LayerNorm(latent_size)
        # K/V/confidence head
        self.proj = nn.Linear(latent_size, head_dim * 2 + 1)
        # Reconstruction head (used only for training; ignored during inference)
        self.reconstruct = nn.Linear(latent_size, chunk_size * hidden_size)

    def forward(self, latent):
        # latent: (b, latent_size)
        x = self.norm(latent)
        hd = self.head_dim
        out = self.proj(x)                              # (b, head_dim*2 + 1)
        k = out[..., :hd]                               # (b, head_dim)
        v = out[..., hd:2*hd]                           # (b, head_dim)
        confidence = torch.sigmoid(out[..., -1:])       # (b, 1)
        # Reconstruction (optional, for training)
        recon = self.reconstruct(x)                     # (b, chunk_size * hidden_size)
        return k, v, confidence, recon
