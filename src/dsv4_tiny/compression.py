"""Compression modules for DSV4-Tiny: CSACompressor and HCACompressor.

Implements Equations 9-12 (CSA) and 20-23 (HCA) from the DSv4 paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DSV4TinyConfig


class CSACompressor(nn.Module):
    """Compressed Sparse Attention (CSA) — fine-grained block compression.

    Compresses m=4 tokens into a single block via dual-stream attention pooling.

    Equations 9-12:
        Ca = H @ W_aKV      (content stream a)
        Cb = H @ W_bKV      (content stream b)
        Za = H @ W_aZ + Ba  (position-aware score stream a)
        Zb = H @ W_bZ + Bb  (position-aware score stream b)
        [Sa; Sb] = Softmax([Za; Zb], dim=-2)   — softmax over block positions
        C_Comp = sum(Sa * Ca, dim=-2) + sum(Sb * Cb, dim=-2)
    """

    def __init__(self, config: DSV4TinyConfig):
        super().__init__()
        d = config.hidden_size            # 1024
        c = config.csa_compressed_dim     # 256
        d_c = config.csa_intermediate     # 512
        g = config.csa_groups             # 4
        m = config.csa_block_size         # 4

        # W_aKV: (d, c * g) — content stream a, projects to grouped compressed dim
        self.W_aKV = nn.Linear(d, c * g, bias=False)

        # W_bKV: (d, d_c) — content stream b, projects to intermediate
        self.W_bKV = nn.Linear(d, d_c, bias=False)

        # W_aZ: (d, 1) — per-position scores for stream a
        self.W_aZ = nn.Linear(d, 1, bias=False)

        # W_bZ: (d, 1) — per-position scores for stream b
        self.W_bZ = nn.Linear(d, 1, bias=False)

        # Position biases Ba, Bb: learnable, shape (1,)
        self.Ba = nn.Parameter(torch.zeros(1))
        self.Bb = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        """Xerox/DSv4-style initialization: N(0, 0.02) for projections, zeros for biases."""
        nn.init.normal_(self.W_aKV.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_bKV.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_aZ.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_bZ.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.Ba)
        nn.init.zeros_(self.Bb)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_mask: torch.Tensor,
        first_block: bool = False,
    ) -> torch.Tensor:
        """Compress a block of KV hidden states.

        Args:
            hidden_states: (batch_size, m, d) — m tokens to compress.
            block_mask: (batch_size, m) — boolean mask for padding within block.
            first_block: If True, applies i=0 boundary: Zb positions padded with -inf,
                Cb zeroed.

        Returns:
            C_Comp: (batch_size, c * g + d_c) — compressed KV representation.
                First c*g dims from Ca stream, last d_c dims from Cb stream.
        """
        b, m_actual, d = hidden_states.shape
        m = 4  # block size (hard-coded; could be config)

        # 1. Compute content streams
        Ca = self.W_aKV(hidden_states)   # (b, m, c*g)
        Cb = self.W_bKV(hidden_states)   # (b, m, d_c)

        # 2. Compute per-position scores + add position biases
        Za = self.W_aZ(hidden_states) + self.Ba  # (b, m, 1)
        Zb = self.W_bZ(hidden_states) + self.Bb  # (b, m, 1)

        # 3. i=0 boundary: first compression block uses only Ca
        if first_block:
            Zb = torch.full_like(Zb, float("-inf"))
            Cb = torch.zeros_like(Cb)

        # 4. Apply block mask: squeeze to (b, m) for softmax
        mask_expanded = block_mask  # (b, m)
        Za = Za.squeeze(-1).masked_fill(~mask_expanded, float("-inf"))  # (b, m)
        Zb = Zb.squeeze(-1).masked_fill(~mask_expanded, float("-inf"))  # (b, m)

        # 5. Concatenate along position dim: (b, 2*m)
        Z_concat = torch.cat([Za, Zb], dim=-1)  # (b, 2*m)
        S = F.softmax(Z_concat, dim=-1)          # (b, 2*m)

        # 6. Split: Sa for stream a, Sb for stream b
        Sa = S[:, :m]     # (b, m) — scores for Ca positions
        Sb = S[:, m:]     # (b, m) — scores for Cb positions

        # 7. Weighted sum: (b, m, 1) * (b, m, c*g) -> sum over positions
        C_comp_a = (Sa.unsqueeze(-1) * Ca).sum(dim=-2)  # (b, c*g)
        C_comp_b = (Sb.unsqueeze(-1) * Cb).sum(dim=-2)  # (b, d_c)

        # 8. Concatenate streams
        C_comp = torch.cat([C_comp_a, C_comp_b], dim=-1)  # (b, c*g + d_c)

        return C_comp


class HCACompressor(nn.Module):
    """Heavy Compressed Attention (HCA) — coarse-grained block compression.

    Compresses m'=128 tokens into a single block via single-stream attention pooling.

    Equations 20-23:
        C = H @ W_KV          (compressed content)
        Z = H @ W_Z + B       (per-position scores)
        S = Softmax(Z + B, dim=-2)
        C_Comp = sum(S * C, dim=-2)
    """

    def __init__(self, config: DSV4TinyConfig):
        super().__init__()
        d = config.hidden_size             # 1024
        c_hca = config.hca_compressed_dim  # 512
        m_prime = config.hca_block_size    # 128

        # W_KV: (d, c_hca) — single-stream compressed KV
        self.W_KV = nn.Linear(d, c_hca, bias=False)

        # W_Z: (d, 1) — per-position scores
        self.W_Z = nn.Linear(d, 1, bias=False)

        # Position bias B: learnable, shape (1,)
        self.B = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.W_KV.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_Z.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.B)

    def forward(
        self,
        hidden_states: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compress a block of KV hidden states (HCA).

        Args:
            hidden_states: (batch_size, m', d) — m' tokens to compress.
            block_mask: (batch_size, m') — boolean mask for padding within block.

        Returns:
            C_Comp: (batch_size, c_hca) — compressed KV representation.
        """
        b, m_prime, d = hidden_states.shape

        # 1. Compute compressed content and scores
        C = self.W_KV(hidden_states)   # (b, m', c_hca)
        Z = self.W_Z(hidden_states) + self.B  # (b, m', 1)

        # 2. Squeeze, apply block mask, softmax over positions
        Z = Z.squeeze(-1).masked_fill(~block_mask, float("-inf"))  # (b, m')
        S = F.softmax(Z, dim=-1)  # (b, m')

        # 3. Weighted sum: (b, m', 1) * (b, m', c_hca) -> sum over positions
        C_comp = (S.unsqueeze(-1) * C).sum(dim=-2)  # (b, c_hca)

        return C_comp
