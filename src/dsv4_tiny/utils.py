"""Utility helpers for DSV4-Tiny: RoPE, block alignment, precision."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor



def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Root Mean Square Layer Normalization.

    Args:
        x: Input tensor (..., hidden_size).
        weight: Learnable weight vector (hidden_size,).
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor of same shape as x.
    """
    dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x_norm = x * torch.rsqrt(variance + eps)
    return (weight * x_norm).to(dtype)

def precompute_freqs_cis(
    dim: int,
    max_position: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Precompute rotary position embedding cos/sin values.

    Returns:
        Tuple (cos, sin) tensors of shape (max_position, dim // 2).
    """
    assert dim % 2 == 0, f"RoPE dim {dim} must be even"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    angles = positions[:, None] * freqs[None, :]  # (max_pos, dim//2)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary_emb(
    x: Tensor,
    freqs_cos: Tensor,
    freqs_sin: Tensor,
    partial_dim: Optional[int] = None,
) -> Tensor:
    """Apply rotary position embeddings.

    Args:
        x: Query or key tensor of shape (..., dim).
        freqs_cos: Precomputed cos values (max_position, dim//2).
        freqs_sin: Precomputed sin values (max_position, dim//2).
        partial_dim: If set, only apply RoPE to the last `partial_dim` dimensions.

    Returns:
        Tensor of same shape as x with RoPE applied.
    """
    if partial_dim is not None and partial_dim < x.shape[-1]:
        full_dim = x.shape[-1]
        static_part = x[..., :full_dim - partial_dim]
        rope_part = x[..., full_dim - partial_dim:]
        rope_part_rotated = _apply_rotary_emb_safe(rope_part, freqs_cos, freqs_sin)
        return torch.cat([static_part, rope_part_rotated], dim=-1)
    return _apply_rotary_emb_safe(x, freqs_cos, freqs_sin)


def _apply_rotary_emb_safe(x: Tensor, freqs_cos: Tensor, freqs_sin: Tensor) -> Tensor:
    """Apply RoPE using standalone cos/sin pairs.

    x can have shapes (T, D), (B, T, D), or (B, T, H, D).
    The seq_len is always found at dim 0 (for 2D) or dim 1 (for 3D+).
    """
    dim = x.shape[-1]
    assert dim % 2 == 0, f"Applied dim {dim} must be even"
    half_dim = dim // 2

    # Identify seq_len dimension: for (T, D) it's dim 0; for (B, T, ...) it's dim 1
    if x.dim() == 2:
        seq_len = x.shape[0]
    else:
        seq_len = x.shape[1]

    # Slice frequencies to match seq_len
    cos = freqs_cos[:seq_len].to(x.dtype)  # (seq_len, half_dim)
    sin = freqs_sin[:seq_len].to(x.dtype)

    # Reshape cos/sin by inserting singleton dims at the front.
    # For (B, T, D): shape (1, seq, half)
    # For (B, T, H, D): shape (1, seq, 1, half)
    # For (T, D): shape (seq, half) — unchanged
    if x.dim() > 2:
        # Add batch dim
        cos = cos.unsqueeze(0)   # (1, seq, half)
        sin = sin.unsqueeze(0)
        # If head dim exists (4D tensor), add another singleton
        if x.dim() == 4:
            cos = cos.unsqueeze(-2)  # (1, seq, 1, half)
            sin = sin.unsqueeze(-2)

    # Split x into two halves
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]

    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    return rotated

def build_attention_mask(
    seq_len: int,
    window_size: Optional[int] = None,
    causal: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Build an attention mask tensor.

    Args:
        seq_len: Sequence length.
        window_size: If set, create a sliding window mask (banded).
        causal: If True, apply causal masking.
        device: Target device.
        dtype: Output dtype.

    Returns:
        Mask tensor of shape (seq_len, seq_len) where True = allowed.
        Uses -inf for masked positions (for add to attention scores).
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    if causal:
        # Causal: each position can attend to itself and earlier positions
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        mask = mask.masked_fill(causal_mask, float("-inf"))
        mask = mask.masked_fill(~causal_mask, 0.0)

    if window_size is not None:
        # Sliding window: only attend to last window_size positions
        # For each query position i, keys j must satisfy i - j < window_size
        dist = torch.arange(seq_len, device=device).unsqueeze(1) - torch.arange(seq_len, device=device).unsqueeze(0)
        window_mask = (dist >= window_size) | (dist < 0)
        mask = mask.masked_fill(window_mask, float("-inf"))
        mask = mask.masked_fill(~window_mask & (mask != float("-inf")), 0.0)

    return mask


def get_linear_schedule_with_cosine_decay(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine learning rate schedule with linear warmup."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_activation(activation: str) -> type[torch.nn.Module]:
    """Get activation by name."""
    if activation == "silu":
        return torch.nn.SiLU
    elif activation == "gelu":
        return torch.nn.GELU
    elif activation == "relu":
        return torch.nn.ReLU
    raise ValueError(f"Unknown activation: {activation}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
