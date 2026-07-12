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

    Every layer now computes SWA + CSA + HCA simultaneously (hierarchical memory).
    """

    def __init__(self, config: DSV4TinyConfig, layer_idx: int, shared_W_DQ: Optional[nn.Linear] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = "hierarchical"

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

        # ── K/V projections (for SWA tier — always present) ──
        self.k_proj = nn.Linear(d, n_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(d, n_kv * head_dim, bias=False)


        # ── Decomposed query (shared with indexer) ──
        c_I = config.indexer_dim            # 128
        n_Ih = config.indexer_heads         # 8
        if config.shared_W_DQ and shared_W_DQ is not None:
            self.W_DQ = shared_W_DQ
        else:
            self.W_DQ = nn.Linear(d, c_I * n_Ih, bias=False)


        # ── Up-projected query for CSA ──
        self.W_UQ = nn.Linear(c_I, head_dim, bias=False)

        # ── Output projection with gate ──
        self.o_proj = nn.Linear(n_h * head_dim, d, bias=False)
        if config.attn_output_gate:
            self.o_gate = nn.Linear(n_h * head_dim, d, bias=False)

        # ── Per-attention-type sink tokens ──
        self.sink_swa = nn.Parameter(torch.randn(1, 1, head_dim) * 0.02)
        self.sink_csa = nn.Parameter(torch.randn(1, 1, head_dim) * 0.02)
        self.sink_hca = nn.Parameter(torch.randn(1, 1, head_dim) * 0.02)

        # ── Learned combination weights for output fusion ──
        self.alpha_swa = nn.Parameter(torch.tensor(1.0))
        self.alpha_csa = nn.Parameter(torch.tensor(1.0))
        self.alpha_hca = nn.Parameter(torch.tensor(1.0))

        # ── RoPE (partial: last rope_dim dims) ──
        self.rope_dim = rope_dim
        self.rope_theta = config.rope_theta

        # Precompute RoPE frequencies (cached on first forward)
        self.register_buffer("_freqs_cos", None, persistent=False)
        self.register_buffer("_freqs_sin", None, persistent=False)

        # Head counts for GQA
        self.n_groups = n_h // n_kv  # 4 heads per KV head

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
        """Forward pass for a single layer — all three tiers simultaneously.

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

        # 3. Run all three tiers
        out_swa = self._forward_swa(hidden_states, q, cache)
        out_csa = self._forward_csa(hidden_states, q, cache)
        out_hca = self._forward_hca(hidden_states, q, cache)

        # 4. Learned weighted combination
        alpha_swa = torch.sigmoid(self.alpha_swa)
        alpha_csa = torch.sigmoid(self.alpha_csa)
        alpha_hca = torch.sigmoid(self.alpha_hca)
        output = alpha_swa * out_swa + alpha_csa * out_csa + alpha_hca * out_hca

        return output

    def _compute_cq(self, hidden_states: torch.Tensor) -> tuple:
        """Compute decomposed query shared by CSA and HCA.

        Returns:
            cQ: (b, seq_len, n_Ih, c_I) decomposed query
        """
        cQ = self.W_DQ(hidden_states)  # (b, seq_len, c_I * n_Ih)
        c_I = self.config.indexer_dim
        n_Ih = self.config.indexer_heads
        cQ = cQ.view(-1, hidden_states.shape[1], n_Ih, c_I)  # (b, seq_len, n_Ih, c_I)
        return cQ

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

            # Also get uncompressed tail for context
            tail_k, tail_v = cache.get_tail_kv(self.layer_idx)
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
                k_full = torch.cat([win_k.unsqueeze(0), k], dim=1)
                v_full = torch.cat([win_v.unsqueeze(0), v], dim=1)
            else:
                k_full = k
                v_full = v
        else:
            k_full = k
            v_full = v

        # Apply attention sink — expand to match n_kv heads
        sink_k = self.sink_swa.unsqueeze(0).expand(b, -1, self.n_kv, -1)  # (b, 1, n_kv, head_dim)
        sink_v = self.sink_swa.unsqueeze(0).expand(b, -1, self.n_kv, -1)
        k_full = torch.cat([sink_k, k_full], dim=1)
        v_full = torch.cat([sink_v, v_full], dim=1)
        # GQA: expand KV heads to match query heads
        k_full = k_full.repeat_interleave(self.n_groups, dim=2)
        v_full = v_full.repeat_interleave(self.n_groups, dim=2)

        # Attention
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.einsum("bqhd,bkhd->bhqk", q, k_full) * scale

        # Causal mask with window constraint
        total_len = k_full.shape[1]
        mask = torch.full((seq_len, total_len), float("-inf"), device=device, dtype=dtype)
        sink_offset = 1  # sink token
        for i in range(seq_len):
            max_k = i + sink_offset
            mask[i, :max_k + 1] = 0.0
            window = self.config.swa_window_size
            min_k = max(0, i + sink_offset - window)
            if min_k > sink_offset:
                mask[i, sink_offset:min_k] = float("-inf")

        attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)
        attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

        # Output
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_probs, v_full)
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
        """Compressed Sparse Attention using Tier 2 cache — per-KV-head GQA."""
        b, seq_len, d = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        head_dim = self.head_dim
        n_h = self.n_h
        kv_heads = self.n_kv
        groups = self.n_groups

        # ── Decomposed query (shared with indexer) ──
        cQ = self._compute_cq(hidden_states)  # (b, seq_len, n_Ih, c_I)

        # ── Up-projected query for CSA ──
        q_up = self.W_UQ(cQ)  # (b, seq_len, n_Ih, head_dim)
        q_csa = q_up

        # Apply partial RoPE to CSA query
        freqs_cos, freqs_sin = self._get_rope_freqs(seq_len, device)
        q_csa = apply_rotary_emb(q_csa, freqs_cos, freqs_sin, partial_dim=self.rope_dim)

        # ── Batched no-cache path (pure sink-token attention, training) ──
        if cache is None:
            # Only sink token available — batched across all positions
            sink_k = self.sink_csa.squeeze(1).expand(kv_heads, -1).unsqueeze(0)  # (1, kv_heads, hd)
            sink_v = self.sink_csa.squeeze(1).expand(kv_heads, -1).unsqueeze(0)
            k_all = sink_k  # (1, kv_heads, hd)
            v_all = sink_v
            k_all_t = k_all.transpose(0, 1)  # (kv_heads, 1, hd)
            v_all_t = v_all.transpose(0, 1)
            scale = 1.0 / math.sqrt(head_dim)
            # Batched over seq_len
            q_gqa = q_csa.view(b, seq_len, kv_heads, groups, head_dim)  # (b, seq_len, kv_heads, groups, hd)
            scores = torch.einsum("blgqd,gkd->blgqk", q_gqa, k_all_t) * scale
            attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            attn_out = torch.einsum("blgqk,gkd->blgqd", attn_probs, v_all_t)
            attn_out = attn_out.reshape(b, seq_len, n_h, head_dim)
            attn_out = attn_out.reshape(b, seq_len, n_h * head_dim)
            out = self.o_proj(attn_out)
            if hasattr(self, "o_gate"):
                gate_val = torch.sigmoid(self.o_gate(attn_out))
                out = out * gate_val
            return out

        output_chunks = []
        for t in range(seq_len):
            # ── Get compressed blocks from Tier 2 cache ──
            query_hidden = hidden_states[:, t]  # (b, d)
            compressed_blocks = cache.get_sparse(self.layer_idx, query_hidden) if cache else []

            # ── Get uncompressed tail (per-KV-head) ──
            tail_k, tail_v = cache.get_tail_kv(self.layer_idx) if cache else (torch.tensor([]), torch.tensor([]))

            # ── Compute attention at this position ──
            q_t = q_csa[:, t]  # (b, n_h, head_dim)

            # Attention sink — per-KV-head
            sink_k = self.sink_csa.squeeze(1)  # (1, head_dim)
            sink_v = self.sink_csa.squeeze(1)
            sink_k = sink_k.expand(kv_heads, -1)
            sink_v = sink_v.expand(kv_heads, -1)

            keys = []
            values = []

            # 1. Sink token — (1, kv_heads, head_dim)
            keys.append(sink_k.unsqueeze(0))
            values.append(sink_v.unsqueeze(0))

            # 2. Compressed blocks (Tier 2) — expand decoder output to kv_heads
            if compressed_blocks:
                for block in compressed_blocks:
                    k_block, v_block, _, _ = cache.decoder(latent)
                    k_block = k_block.expand(1, kv_heads, -1)  # (1, kv_heads, head_dim)
                    v_block = v_block.expand(1, kv_heads, -1)
                    keys.append(k_block)
                    values.append(v_block)

            # 3. Uncompressed tail — already (tail_len, kv_heads, head_dim)
            if tail_k.numel() > 0 and tail_k.shape[0] > 0:
                if tail_k.dim() == 3 and tail_k.shape[1] == kv_heads:
                    keys.append(tail_k)
                    values.append(tail_v)
                elif tail_k.dim() == 2 and tail_k.shape[-1] == head_dim:
                    keys.append(tail_k.unsqueeze(1).expand(-1, kv_heads, -1))
                    values.append(tail_v.unsqueeze(1).expand(-1, kv_heads, -1))
                else:
                    if tail_k.shape[-1] >= head_dim:
                        tk = tail_k[:, :head_dim]
                        tv = tail_v[:, :head_dim]
                    else:
                        repeats = (head_dim + tail_k.shape[-1] - 1) // tail_k.shape[-1]
                        tk = tail_k.repeat(1, repeats)[:, :head_dim]
                        tv = tail_v.repeat(1, repeats)[:, :head_dim]
                    keys.append(tk.unsqueeze(1).expand(-1, kv_heads, -1))
                    values.append(tv.unsqueeze(1).expand(-1, kv_heads, -1))

            k_all = torch.cat(keys, dim=0)   # (n_keys, kv_heads, head_dim)
            v_all = torch.cat(values, dim=0)

            # Transpose K/V for GQA einsum: (kv_heads, n_keys, head_dim)
            k_all_t = k_all.transpose(0, 1)
            v_all_t = v_all.transpose(0, 1)

            # GQA attention: per-kv-head groups
            scale = 1.0 / math.sqrt(head_dim)
            q_gqa = q_t.view(b, kv_heads, groups, head_dim)
            scores = torch.einsum("bgqd,gkd->bgqk", q_gqa, k_all_t) * scale

            attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            attn_out = torch.einsum("bgqk,gkd->bgqd", attn_probs, v_all_t)
            attn_out = attn_out.reshape(b, n_h, head_dim)
            attn_out = attn_out.reshape(b, n_h * head_dim)

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
        """Heavy Compressed Attention using Tier 3 cache — per-KV-head GQA."""


        b, seq_len, d = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        head_dim = self.head_dim
        n_h = self.n_h
        kv_heads = self.n_kv
        groups = self.n_groups

        # Use standard Q as the attention query (n_h, head_dim)
        q_attn = q

        # Decomposed query (for future indexer scoring; not up-projected for HCA)
        _ = self._compute_cq(hidden_states)

        freqs_cos, freqs_sin = self._get_rope_freqs(seq_len, device)
        q_attn = apply_rotary_emb(q_attn, freqs_cos, freqs_sin, partial_dim=self.rope_dim)
        # ── Batched no-cache path (pure sink-token attention, training) ──
        if cache is None:
            sink_k = self.sink_hca.squeeze(1).expand(kv_heads, -1).unsqueeze(0)
            sink_v = self.sink_hca.squeeze(1).expand(kv_heads, -1).unsqueeze(0)
            k_all = sink_k
            v_all = sink_v
            k_all_t = k_all.transpose(0, 1)
            v_all_t = v_all.transpose(0, 1)
            scale = 1.0 / math.sqrt(head_dim)
            q_gqa = q_attn.view(b, seq_len, kv_heads, groups, head_dim)
            scores = torch.einsum("blgqd,gkd->blgqk", q_gqa, k_all_t) * scale
            attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            attn_out = torch.einsum("blgqk,gkd->blgqd", attn_probs, v_all_t)
            attn_out = attn_out.reshape(b, seq_len, n_h, head_dim)
            attn_out = attn_out.reshape(b, seq_len, n_h * head_dim)
            out = self.o_proj(attn_out)
            if hasattr(self, "o_gate"):
                gate_val = torch.sigmoid(self.o_gate(attn_out))
                out = out * gate_val
            return out

        output_chunks = []
        for t in range(seq_len):
            # ── Get all compressed blocks from Tier 3 cache ──
            blocks = cache.get_dense(self.layer_idx) if cache else []

            # ── Get uncompressed tail ──
            tail_k, tail_v = cache.get_tail_kv(self.layer_idx) if cache else (torch.tensor([]), torch.tensor([]))

            # ── HCA query at position t ──
            q_t = q_attn[:, t]  # (b, n_h, head_dim)

            # Attention sink — per-KV-head
            sink_k = self.sink_hca.squeeze(1)
            sink_v = self.sink_hca.squeeze(1)
            sink_k = sink_k.expand(kv_heads, -1)
            sink_v = sink_v.expand(kv_heads, -1)

            keys = []
            values = []

            # 1. Sink token — (1, kv_heads, head_dim)
            keys.append(sink_k.unsqueeze(0))
            values.append(sink_v.unsqueeze(0))

            # 2. Compressed blocks (Tier 3) — expand decoder output to kv_heads
            if blocks:
                for block in blocks:
                    k_block, v_block, _, _ = cache.decoder(latent)
                    k_block = k_block.expand(1, kv_heads, -1)
                    v_block = v_block.expand(1, kv_heads, -1)
                    keys.append(k_block)
                    values.append(v_block)

            # 3. Uncompressed tail — already (tail_len, kv_heads, head_dim)
            if tail_k.numel() > 0 and tail_k.shape[0] > 0:
                if tail_k.dim() == 3 and tail_k.shape[1] == kv_heads:
                    keys.append(tail_k)
                    values.append(tail_v)
                elif tail_k.dim() == 2 and tail_k.shape[-1] == head_dim:
                    keys.append(tail_k.unsqueeze(1).expand(-1, kv_heads, -1))
                    values.append(tail_v.unsqueeze(1).expand(-1, kv_heads, -1))
                else:
                    if tail_k.shape[-1] >= head_dim:
                        tk = tail_k[:, :head_dim]
                        tv = tail_v[:, :head_dim]
                    else:
                        repeats = (head_dim + tail_k.shape[-1] - 1) // tail_k.shape[-1]
                        tk = tail_k.repeat(1, repeats)[:, :head_dim]
                        tv = tail_v.repeat(1, repeats)[:, :head_dim]
                    keys.append(tk.unsqueeze(1).expand(-1, kv_heads, -1))
                    values.append(tv.unsqueeze(1).expand(-1, kv_heads, -1))

            k_all = torch.cat(keys, dim=0)
            v_all = torch.cat(values, dim=0)

            k_all_t = k_all.transpose(0, 1)
            v_all_t = v_all.transpose(0, 1)

            scale = 1.0 / math.sqrt(head_dim)
            q_gqa = q_t.view(b, kv_heads, groups, head_dim)
            scores = torch.einsum("bgqd,gkd->bgqk", q_gqa, k_all_t) * scale

            attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
            attn_out = torch.einsum("bgqk,gkd->bgqd", attn_probs, v_all_t)
            attn_out = attn_out.reshape(b, n_h, head_dim)
            attn_out = attn_out.reshape(b, n_h * head_dim)

            out = self.o_proj(attn_out)
            if hasattr(self, "o_gate"):
                gate_val = torch.sigmoid(self.o_gate(attn_out))
                out = out * gate_val

            output_chunks.append(out)

        return torch.stack(output_chunks, dim=1)

