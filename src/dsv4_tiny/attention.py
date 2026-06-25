"""DSV4Attention — three-tier heterogeneous attention.

Each DSV4Attention layer supports SWA (sliding window), CSA (compressed sparse),
or HCA (heavy compressed) attention based on layer type assignment.

Architecture:
    - Q projection from base model weights
    - KV either from cache (CSA/HCA compressed) or computed (SWA)
    - Decomposed query shared with LightningIndexer
    - MQA-style attention (all heads share compressed KV)
    - Grouped output projection with gate
    - Attention sink token (Eq 27)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DSV4TinyConfig
from .cache import DSV4Cache
from .utils import apply_rotary_emb, precompute_freqs_cis


class DSV4Attention(nn.Module):
    """Single layer of DSV4 tiered attention.

    Handles SWA, CSA, and HCA attention types based on layer index.
    """

    def __init__(self, config: DSV4TinyConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_type(layer_idx)

        d = config.hidden_size               # 1024
        n_h = config.num_attention_heads     # 8
        n_kv = config.num_key_value_heads    # 2
        head_dim = config.head_dim           # 256
        rope_dim = config.rope_dim           # 64
        g = config.csa_groups                # 4

        self.n_h = n_h
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.d = d

        # ── Q projection (from base model) ──
        self.q_proj = nn.Linear(d, n_h * head_dim, bias=False)

        # ── K/V projections (used only for SWA layers) ──
        if self.layer_type == "swa":
            self.k_proj = nn.Linear(d, n_kv * head_dim, bias=False)
            self.v_proj = nn.Linear(d, n_kv * head_dim, bias=False)
        else:
            # For compressed layers, KV comes from cache via learned decompression
            self.kv_decompress = nn.Linear(
                self._compressed_kv_dim(), head_dim * 2, bias=False
            )

        # ── Decomposed query (shared with indexer) ──
        c_I = config.indexer_dim            # 128
        n_Ih = config.indexer_heads         # 8
        self.W_DQ = nn.Linear(d, c_I * n_Ih, bias=False)

        # ── Up-projected query for CSA ──
        if self.layer_type == "csa":
            # Per-head up-projection: c_I -> head_dim
            self.W_UQ = nn.Linear(c_I, head_dim, bias=False)

        # ── Output projection with gate ──
        self.o_proj = nn.Linear(n_h * head_dim, d, bias=False)
        if config.attn_output_gate:
            # Gate: element-wise gating of output
            self.o_gate = nn.Linear(n_h * head_dim, d, bias=False)

        # ── Attention sink token (Eq 27) ──
        # Shape: (1, 1, head_dim) — shared KV head for all layers
        self.sink_token = nn.Parameter(torch.randn(1, 1, head_dim) * 0.02)



        # ── RoPE (partial: last rope_dim dims) ──
        self.rope_dim = rope_dim
        self.rope_theta = config.rope_theta

        # Precompute RoPE frequencies (cached on first forward)
        self.register_buffer("_freqs_cos", None, persistent=False)
        self.register_buffer("_freqs_sin", None, persistent=False)

        # Head counts for GQA
        self.n_groups = n_h // n_kv  # 4 heads per KV head

    def _compressed_kv_dim(self) -> int:
        """Return the dim of compressed KV for this layer type."""
        cfg = self.config
        if self.layer_type == "csa":
            # CSA: all sub-blocks concatenated: (c*g + d_c) * (block_size / csa_block_size)
            sub_block_dim = cfg.csa_compressed_dim * cfg.csa_groups + cfg.csa_intermediate
            num_sub_blocks = cfg.block_alignment // cfg.csa_block_size  # 128/4 = 32
            return sub_block_dim * num_sub_blocks
        elif self.layer_type == "hca":
            # HCA: c_hca (one block per 128-token aligned block)
            return cfg.hca_compressed_dim
        return 0

    def _get_rope_freqs(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Get or precompute RoPE frequencies (cos, sin)."""
        if self._freqs_cos is None or self._freqs_cos.shape[0] < seq_len:
            cos, sin = precompute_freqs_cis(
                self.rope_dim, self.config.max_position_embeddings,
                theta=self.rope_theta, device=device,
            )
            self.register_buffer("_freqs_cos", cos, persistent=False)
            self.register_buffer("_freqs_sin", sin, persistent=False)
        return self._freqs_cos[:seq_len], self._freqs_sin[:seq_len]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[DSV4Cache] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a single layer.

        Args:
            hidden_states: (batch_size, seq_len, d)
            cache: DSV4Cache instance (must be set during inference/generation).
            position_ids: (batch_size, seq_len) — optional position IDs.

        Returns:
            output: (batch_size, seq_len, d)
        """
        b, seq_len, d = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1. Q projection
        q = self.q_proj(hidden_states)  # (b, seq_len, n_h * head_dim)
        q = q.view(b, seq_len, self.n_h, self.head_dim)

        # 2. Apply partial RoPE to Q
        freqs_cos, freqs_sin = self._get_rope_freqs(seq_len, device)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin, partial_dim=self.rope_dim)

        if self.layer_type == "swa":
            output = self._forward_swa(hidden_states, q, cache)
        elif self.layer_type == "csa":
            output = self._forward_csa(hidden_states, q, cache)
        elif self.layer_type == "hca":
            output = self._forward_hca(hidden_states, q, cache)
        else:
            raise ValueError(f"Unknown layer type: {self.layer_type}")

        return output

    def _forward_swa(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        cache: Optional[DSV4Cache],
    ) -> torch.Tensor:
        """Sliding window attention using uncompressed KV from cache."""
        b, seq_len, d = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        head_dim = self.head_dim
        n_kv = self.n_kv

        # Get KV from the cache: state window + tail
        if cache is not None:
            # 1. Compute K and V for this sequence
            k = self.k_proj(hidden_states)  # (b, seq_len, n_kv * head_dim)
            v = self.v_proj(hidden_states)  # (b, seq_len, n_kv * head_dim)

            k = k.view(b, seq_len, n_kv, head_dim)
            v = v.view(b, seq_len, n_kv, head_dim)

            # Apply RoPE to K
            freqs_cos, freqs_sin = self._get_rope_freqs(seq_len, device)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin, partial_dim=self.rope_dim)

            # Push KV pairs to cache (also pass hidden state for compression)
            for t in range(seq_len):
                hs = hidden_states[:, t].squeeze(0)  # (d,)
                cache.push(
                    self.layer_idx,
                    k[:, t].squeeze(0),
                    v[:, t].squeeze(0),
                    hidden_state=hs,
                )

            # Combine with cached state window
            win_k, win_v = cache.get_state_window(self.layer_idx)
            # win_k/v: (win_len, n_kv, head_dim)
            if win_k.numel() > 0 and win_k.shape[0] > 0:
                # Remove the current sequence's contribution from window
                # (since cache already updated, window includes the current seq)
                # The window already has the sliding KV; we use the current
                # seq's K/V for the actual attention computation.
                pass

            # Also get uncompressed tail for context
            tail_k, tail_v = cache.get_tail_kv(self.layer_idx)
            # tail_k/v: (tail_len, n_kv, head_dim)
        else:
            # No cache — compute K/V directly for this sequence
            k = self.k_proj(hidden_states).view(b, seq_len, n_kv, head_dim)
            v = self.v_proj(hidden_states).view(b, seq_len, n_kv, head_dim)
            freqs_cos, freqs_sin = self._get_rope_freqs(seq_len, device)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin, partial_dim=self.rope_dim)
            tail_k = torch.tensor([], device=device)
            tail_v = torch.tensor([], device=device)

        # Build full key/value for attention
        if cache is not None:
            win_k, win_v = cache.get_state_window(self.layer_idx)
            if win_k.numel() > 0:
                k_full = torch.cat([win_k.unsqueeze(0), k], dim=1)   # (b, win_len + seq_len, n_kv, head_dim)
                v_full = torch.cat([win_v.unsqueeze(0), v], dim=1)
            else:
                k_full = k
                v_full = v
        else:
            k_full = k
            v_full = v

        # Apply attention sink — expand to match n_kv heads
        sink_k = self.sink_token.unsqueeze(0).expand(b, -1, self.n_kv, -1)  # (b, 1, n_kv, head_dim)
        sink_v = self.sink_token.unsqueeze(0).expand(b, -1, self.n_kv, -1)
        k_full = torch.cat([sink_k, k_full], dim=1)
        v_full = torch.cat([sink_v, v_full], dim=1)
        # GQA: expand KV heads to match query heads
        # Q: (b, seq_len, n_h, head_dim), K/V: (b, total_len, n_kv, head_dim)
        k_full = k_full.repeat_interleave(self.n_groups, dim=2)  # (b, total_len, n_h, head_dim)
        v_full = v_full.repeat_interleave(self.n_groups, dim=2)

        # Attention
        scale = 1.0 / math.sqrt(head_dim)
        # (b, n_h, seq_len, total_len)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k_full) * scale

        # Causal mask with window constraint
        total_len = k_full.shape[1]
        mask = torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype)
        sink_offset = 1  # sink token
        for i in range(seq_len):
            # Causal: can attend to positions <= current + sink
            max_k = i + sink_offset
            mask[i, :max_k + 1] = 0.0
            # Window: only last window_size positions (excluding sink)
            window = self.config.swa_window_size
            min_k = max(0, i + sink_offset - window)
            if min_k > sink_offset:
                mask[i, sink_offset:min_k] = float("-inf")

        attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

        # Output
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v_full)  # (b, seq_len, n_h, head_dim)
        attn_output = attn_output.reshape(b, seq_len, self.n_h * head_dim)

        # Output projection with gate
        output = self.o_proj(attn_output)
        if hasattr(self, "o_gate"):
            gate = torch.sigmoid(self.o_gate(attn_output))
            output = output * gate

        return output

    def _forward_csa(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        cache: Optional[DSV4Cache],
    ) -> torch.Tensor:
        """Compressed Sparse Attention using Tier 2 cache."""
        b, seq_len, d = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        head_dim = self.head_dim
        n_h = self.n_h
        n_kv = self.n_kv

        # ── Decomposed query (shared with indexer) ──
        cQ = self.W_DQ(hidden_states)  # (b, seq_len, c_I * n_Ih)
        c_I = self.config.indexer_dim
        n_Ih = self.config.indexer_heads
        cQ = cQ.view(b, seq_len, n_Ih, c_I)  # (b, seq_len, n_Ih, c_I)

        # ── Up-projected query for CSA ──
        q_up = self.W_UQ(cQ)  # (b, seq_len, n_Ih, head_dim)
        # Rename for clarity: q_up is the CSA query, same dim as q
        q_csa = q_up

        # Apply partial RoPE to CSA query
        freqs_cos, freqs_sin = self._get_rope_freqs(seq_len, device)
        q_csa = apply_rotary_emb(q_csa, freqs_cos, freqs_sin, partial_dim=self.rope_dim)

        # Also keep the standard Q for the uncompressed tail portion
        # Standard Q: use q directly for the tail

        output_chunks = []
        for t in range(seq_len):
            # Push hidden state to cache (for future compression)
            if cache is not None:
                cache.push(
                    self.layer_idx,
                    hidden_states[:, t].squeeze(0),  # key proxy
                    hidden_states[:, t].squeeze(0),  # value proxy
                    hidden_state=hidden_states[:, t].squeeze(0),
                )

            # ── Get compressed blocks from Tier 2 cache ──
            query_hidden = hidden_states[:, t]  # (b, d)
            compressed_blocks = cache.get_sparse(self.layer_idx, query_hidden) if cache else []

            # ── Get uncompressed tail ──
            tail_k, tail_v = cache.get_tail_kv(self.layer_idx) if cache else (torch.tensor([]), torch.tensor([]))

            # ── Compute attention at this position ──
            # CSA query at position t
            q_t = q_csa[:, t]  # (b, n_h, head_dim) — q_up per head

            # Attention sink — squeeze to (1, head_dim) to match other keys
            sink_k = self.sink_token.squeeze(1)  # (1, head_dim)
            sink_v = self.sink_token.squeeze(1)

            # Initialize key/value lists
            keys = []
            values = []

            # 1. Sink token
            keys.append(sink_k)
            values.append(sink_v)


            # 2. Compressed blocks (Tier 2)
            if compressed_blocks:
                # Decompress each block
                for block in compressed_blocks:
                    # block.compressed_kv: (compressed_dim,)
                    decomp = self.kv_decompress(block.compressed_kv.unsqueeze(0))  # (1, head_dim * 2)
                    k_block = decomp[:, :head_dim]     # (1, head_dim)
                    v_block = decomp[:, head_dim:]     # (1, head_dim)
                    keys.append(k_block)
                    values.append(v_block)

            # 3. Uncompressed tail (handle both (tail_len, kv_heads, head_dim) and (tail_len, hidden_size))
            if tail_k.numel() > 0 and tail_k.shape[0] > 0:
                if tail_k.dim() == 3:
                    # KV format: (tail_len, kv_heads, head_dim) -> average heads
                    tk = tail_k.mean(dim=1)  # (tail_len, head_dim)
                    tv = tail_v.mean(dim=1)
                elif tail_k.dim() == 2 and tail_k.shape[-1] == self.head_dim:
                    tk = tail_k  # (tail_len, head_dim)
                    tv = tail_v
                else:
                    # Hidden state format: (tail_len, hidden_size) -> project to head_dim
                    # Use a simple learned projection if available, else mean pool
                    if tail_k.shape[-1] >= self.head_dim:
                        tk = tail_k[:, :self.head_dim]  # truncate
                        tv = tail_v[:, :self.head_dim]
                    else:
                        # Pad if too small (shouldn't happen with d=1024, head_dim=256)
                        repeats = (self.head_dim + tail_k.shape[-1] - 1) // tail_k.shape[-1]
                        tk = tail_k.repeat(1, repeats)[:, :self.head_dim]
                        tv = tail_v.repeat(1, repeats)[:, :self.head_dim]
                keys.append(tk)
                values.append(tv)

            k_all = torch.cat(keys, dim=0).unsqueeze(0)   # (1, num_keys, head_dim)
            v_all = torch.cat(values, dim=0).unsqueeze(0)  # (1, num_keys, head_dim)

            # MQA: all query heads share K/V
            # Q: (b, n_h, hd), K: (num_keys, hd) -> scores: (b, n_h, num_keys)
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.einsum("bnd,kd->bnk", q_t, k_all.squeeze(0)) * scale

            attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            # (b, n_h, num_keys) @ (num_keys, hd) -> (b, n_h, hd)
            attn_out = torch.einsum("bnk,kd->bnd", attn_probs, v_all.squeeze(0))

            attn_out = attn_out.reshape(b, n_h * head_dim)

            # Output projection with gate
            out = self.o_proj(attn_out)
            if hasattr(self, "o_gate"):
                gate_val = torch.sigmoid(self.o_gate(attn_out))
                out = out * gate_val

            output_chunks.append(out)

        return torch.stack(output_chunks, dim=1)  # (b, seq_len, d)

    def _forward_hca(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        cache: Optional[DSV4Cache],
    ) -> torch.Tensor:
        """Heavy Compressed Attention using Tier 3 cache."""
        b, seq_len, d = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        head_dim = self.head_dim
        n_h = self.n_h


        # For HCA, use the decomposed query (no up-projection)
        cQ = self.W_DQ(hidden_states)  # (b, seq_len, c_I * n_Ih)
        c_I = self.config.indexer_dim
        n_Ih = self.config.indexer_heads
        q_hca = cQ.view(b, seq_len, n_Ih, c_I)  # (b, seq_len, n_Ih, c_I)

        # For HCA, query dim (c_I=128) differs from head_dim (256).
        # We need a projection from c_I to head_dim for the attention computation.
        # Use Q from standard projection as the attention query.
        # The decomposed query cQ is used for scoring blocks.
        q_attn = q  # Use standard Q: (b, seq_len, n_h, head_dim)

        freqs_cos, freqs_sin = self._get_rope_freqs(seq_len, device)
        q_attn = apply_rotary_emb(q_attn, freqs_cos, freqs_sin, partial_dim=self.rope_dim)

        output_chunks = []
        for t in range(seq_len):
            # Push hidden state to cache (for future compression)
            if cache is not None:
                cache.push(
                    self.layer_idx,
                    hidden_states[:, t].squeeze(0),  # key proxy
                    hidden_states[:, t].squeeze(0),  # value proxy
                    hidden_state=hidden_states[:, t].squeeze(0),
                )

            # ── Get all compressed blocks from Tier 3 cache ──
            blocks = cache.get_dense(self.layer_idx) if cache else []

            # ── Get uncompressed tail ──
            tail_k, tail_v = cache.get_tail_kv(self.layer_idx) if cache else (torch.tensor([]), torch.tensor([]))

            # ── HCA query at position t ──
            q_t = q_attn[:, t]  # (b, n_h, head_dim)

            # Attention sink — squeeze to (1, head_dim) to match other keys
            sink_k = self.sink_token.squeeze(1)  # (1, head_dim)
            sink_v = self.sink_token.squeeze(1)


            keys = []
            values = []

            # 1. Sink token
            keys.append(sink_k)
            values.append(sink_v)

            # 2. Compressed blocks (Tier 3) — all dense
            if blocks:
                for block in blocks:
                    decomp = self.kv_decompress(block.compressed_kv.unsqueeze(0))  # (1, head_dim * 2)
                    k_block = decomp[:, :head_dim]
                    v_block = decomp[:, head_dim:]
                    keys.append(k_block)
                    values.append(v_block)

            # 3. Uncompressed tail (handle both KV and hidden state formats)
            if tail_k.numel() > 0 and tail_k.shape[0] > 0:
                if tail_k.dim() == 3:
                    tk = tail_k.mean(dim=1)
                    tv = tail_v.mean(dim=1)
                elif tail_k.dim() == 2 and tail_k.shape[-1] == self.head_dim:
                    tk = tail_k
                    tv = tail_v
                else:
                    if tail_k.shape[-1] >= self.head_dim:
                        tk = tail_k[:, :self.head_dim]
                        tv = tail_v[:, :self.head_dim]
                    else:
                        repeats = (self.head_dim + tail_k.shape[-1] - 1) // tail_k.shape[-1]
                        tk = tail_k.repeat(1, repeats)[:, :self.head_dim]
                        tv = tail_v.repeat(1, repeats)[:, :self.head_dim]
                keys.append(tk)
                values.append(tv)

            k_all = torch.cat(keys, dim=0).unsqueeze(0)
            v_all = torch.cat(values, dim=0).unsqueeze(0)

            # MQA attention
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.einsum("bnd,kd->bnk", q_t, k_all.squeeze(0)) * scale

            attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            attn_out = torch.einsum("bnk,kd->bnd", attn_probs, v_all.squeeze(0))

            attn_out = attn_out.reshape(b, n_h * head_dim)

            out = self.o_proj(attn_out)
            if hasattr(self, "o_gate"):
                gate_val = torch.sigmoid(self.o_gate(attn_out))
                out = out * gate_val

            output_chunks.append(out)

        return torch.stack(output_chunks, dim=1)
