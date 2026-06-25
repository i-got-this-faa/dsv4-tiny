"""DSV4-Tiny three-tier heterogeneous KV cache.

Tiers:
  Tier 1 — StateCache: SWA buffer (ring buffer of last n_win KV pairs) +
                        uncompressed tail (pending compression).
  Tier 2 — CSA compressed blocks: fine-grained block cache + LightningIndexer.
  Tier 3 — HCA compressed blocks: coarse-grained block cache (dense retrieval).

Block alignment:
    Block size = lcm(m, m') = lcm(4, 128) = 128 tokens.
    - Each HCA block = 1 aligned block (128 tokens).
    - Each CSA block spans 32 sub-blocks (128/4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import DSV4TinyConfig
from .compression import CSACompressor, HCACompressor
from .indexer import LightningIndexer


@dataclass
class BlockAlignment:
    """Block alignment metadata."""
    block_size: int         # 128 (lcm of CSA and HCA block sizes)
    csa_sub_blocks: int     # 32 (128 / 4)
    hca_blocks: int         # 1 (128 / 128)


@dataclass
class CompressedBlock:
    """A compressed KV block stored in Tier 2 or Tier 3."""
    block_id: int
    compressed_kv: torch.Tensor           # compressed representation
    indexer_key: Optional[torch.Tensor] = None  # KI_Comp for LightnightIndexer (CSA only)


class StateBuffer:
    """Fixed-size ring buffer for SWA layer uncompressed KV pairs."""

    def __init__(self, window_size: int, dtype: torch.dtype = torch.float16):
        self.window_size = window_size
        self.dtype = dtype
        self.clear()

    def clear(self) -> None:
        self.keys = []
        self.values = []
        self._position = 0

    @property
    def size(self) -> int:
        return len(self.keys)

    def push(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append a single KV pair. If buffer exceeds window, drop oldest."""
        self.keys.append(key)
        self.values.append(value)
        if len(self.keys) > self.window_size:
            self.keys.pop(0)
            self.values.pop(0)

    def get_buffer(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (keys, values) tensors of the current window content.

        Returns:
            keys: (window_size, kv_heads, head_dim) or smaller if not full.
            values: Same shape as keys.
        """
        if not self.keys:
            return torch.tensor([]), torch.tensor([])
        return torch.stack(self.keys, dim=0), torch.stack(self.values, dim=0)


class DSV4Cache(nn.Module):
    """Three-tier heterogeneous KV cache.

    Manages KV storage across SWA, CSA, and HCA layers.
    Compression modules are registered as sub-modules for training.
    """

    def __init__(self, config: DSV4TinyConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Block alignment
        self.block_size = config.block_alignment  # 128
        self.csa_block_size = config.csa_block_size  # 4
        self.hca_block_size = config.hca_block_size  # 128
        self.csa_sub_blocks = self.block_size // self.csa_block_size  # 32

        # ── Compression modules ──
        self.csa_compressors = nn.ModuleList([
            CSACompressor(config) for _ in range(config.num_hidden_layers)
        ])
        self.hca_compressors = nn.ModuleList([
            HCACompressor(config) for _ in range(config.num_hidden_layers)
        ])
        self.indexer = LightningIndexer(config)

        # ── Per-layer state ──
        # Tier 1: StateCache — one StateBuffer per layer for SWA contexts
        self.swa_buffers: list[StateBuffer] = [
            StateBuffer(config.swa_window_size) for _ in range(self.num_layers)
        ]

        # Tier 1: Uncompressed tail — KV pairs pending compression
        # Stored as lists of (key, value) tuples per layer
        self.tails: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(self.num_layers)
        ]

        # Tier 2: CSA compressed blocks — list of CompressedBlock per CSA layer
        self.csa_cache: list[list[CompressedBlock]] = [[] for _ in range(self.num_layers)]

        # Tier 3: HCA compressed blocks — list of CompressedBlock per HCA layer
        self.hca_cache: list[list[CompressedBlock]] = [[] for _ in range(self.num_layers)]

        # Per-layer cumulative token count (for block_id assignment)
        self.token_counts: list[int] = [0] * self.num_layers

        # Indexer key dim
        self.indexer_key_dim = config.indexer_dim * config.indexer_heads  # 128 * 8 = 1024

        # First-block tracking for i=0 boundary
        self._layer_first_compression: list[bool] = [True] * self.num_layers

    def reset(self) -> None:
        """Clear all cached state (call at start of each sequence)."""
        for buf in self.swa_buffers:
            buf.clear()
        for i in range(self.num_layers):
            self.tails[i].clear()
            self.csa_cache[i].clear()
            self.hca_cache[i].clear()
            self.token_counts[i] = 0
            self._layer_first_compression[i] = True

    def push(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor,
             hidden_state: Optional[torch.Tensor] = None) -> None:
        """Push a single KV pair to the cache for the given layer.

        Args:
            layer_idx: Which layer the KV pair belongs to.
            key: (kv_heads, head_dim) tensor.
            value: Same shape as key.
            hidden_state: (hidden_size,) — optional hidden state for compression.
        """
        layer_type = self.config.layer_type(layer_idx)
        self.token_counts[layer_idx] += 1

        # Add to uncompressed tail (include hidden state for compression)
        self.tails[layer_idx].append((key, value, hidden_state))

        # Always update SWA buffer (needed by SWA layers)
        self.swa_buffers[layer_idx].push(key, value)

        # Check if tail reaches block alignment boundary
        if len(self.tails[layer_idx]) >= self.block_size:
            self._compress_tail(layer_idx, layer_type)


    def _compress_tail(self, layer_idx: int, layer_type: str) -> None:
        """Compress the oldest block of KV pairs from the tail."""
        # Pop oldest block_size pairs
        block_items = self.tails[layer_idx][:self.block_size]
        self.tails[layer_idx] = self.tails[layer_idx][self.block_size:]

        # Extract hidden states for compression (prefer hidden_state, fallback to key mean)
        hidden_states = []
        has_real_hs = False
        for item in block_items:
            hs = item[2] if len(item) > 2 and item[2] is not None else None
            if hs is None:
                # Fallback: use key mean as proxy
                hs = item[0].mean(dim=0)  # (head_dim,) or (hidden_size,)
                if hs.shape[-1] != self.hidden_size:
                    # Pad to hidden_size by repeating (crude but functional)
                    repeats = (self.hidden_size + hs.shape[-1] - 1) // hs.shape[-1]
                    hs = hs.repeat(repeats)[:self.hidden_size]
            else:
                has_real_hs = True
            hidden_states.append(hs)
        hidden_proxy = torch.stack(hidden_states, dim=0)  # (block_size, hidden_size)

        block_id = self.token_counts[layer_idx] // self.block_size
        first_block = self._layer_first_compression[layer_idx]
        if first_block:
            self._layer_first_compression[layer_idx] = False

        if layer_type == "csa":
            self._compress_csa(layer_idx, block_id, hidden_proxy, first_block)
        elif layer_type == "hca":
            self._compress_hca(layer_idx, block_id, hidden_proxy, first_block)

    def _compress_csa(
        self, layer_idx: int, block_id: int,
        hidden_proxy: torch.Tensor, first_block: bool,
    ) -> None:
        """Compress via CSA compressor and store in Tier 2."""
        compressor = self.csa_compressors[layer_idx]
        # Process each CSA sub-block within the aligned block
        compressed_chunks = []
        for i in range(self.csa_sub_blocks):
            start = i * self.csa_block_size
            end = start + self.csa_block_size
            chunk = hidden_proxy[start:end].unsqueeze(0)  # (1, m, d)
            block_mask = torch.ones(1, self.csa_block_size, dtype=torch.bool, device=hidden_proxy.device)
            c_comp = compressor(chunk, block_mask, first_block=(first_block and i == 0))
            compressed_chunks.append(c_comp)

        # Concatenate sub-block compressions
        compressed_kv = torch.cat(compressed_chunks, dim=-1)  # (1, c*g + d_c)

        # Indexer key: project CSA compressed dim to indexer dim
        csa_compressed_dim = self.config.csa_compressed_dim * self.config.csa_groups + self.config.csa_intermediate
        indexer_key_dim = self.config.indexer_dim * self.config.indexer_heads
        if not hasattr(self, "_idx_proj") or self._idx_proj is None:
            self._idx_proj = nn.Linear(csa_compressed_dim, indexer_key_dim, bias=False).to(
                compressed_kv.device, dtype=compressed_kv.dtype
            )
        # Use the FIRST sub-block's compressed KV as the indexer key
        first_sub_block = compressed_chunks[0]  # (1, c*g + d_c)
        indexer_key = self._idx_proj(first_sub_block)  # (1, indexer_key_dim)

        block = CompressedBlock(
            block_id=block_id,
            compressed_kv=compressed_kv.squeeze(0),
            indexer_key=indexer_key.squeeze(0),
        )
        self.csa_cache[layer_idx].append(block)

    def _compress_hca(
        self, layer_idx: int, block_id: int,
        hidden_proxy: torch.Tensor, first_block: bool,
    ) -> None:
        """Compress via HCA compressor and store in Tier 3."""
        compressor = self.hca_compressors[layer_idx]
        hidden_2d = hidden_proxy.unsqueeze(0)  # (1, block_size, d)
        block_mask = torch.ones(1, self.hca_block_size, dtype=torch.bool, device=hidden_proxy.device)
        compressed_kv = compressor(hidden_2d, block_mask)  # (1, c_hca)

        block = CompressedBlock(
            block_id=block_id,
            compressed_kv=compressed_kv.squeeze(0),
        )
        self.hca_cache[layer_idx].append(block)

    def get_state_window(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tier 1: Return the SWA buffer content for sliding window attention.

        Returns:
            (keys, values) tensors of shape (window_size, kv_heads, head_dim).
        """
        return self.swa_buffers[layer_idx].get_buffer()

    def get_sparse(self, layer_idx: int, query_hidden: torch.Tensor) -> list[CompressedBlock]:
        """Tier 2: Retrieve top-k compressed blocks via LightningIndexer.

        Args:
            layer_idx: CSA layer index.
            query_hidden: (batch_size, d) — current token's hidden state.

        Returns:
            List of top-k CompressedBlock objects.
        """
        blocks = self.csa_cache[layer_idx]
        if not blocks:
            return []

        num_blocks = len(blocks)
        # Stack compressed keys for indexer
        if blocks[0].indexer_key is None:
            return blocks  # Fall back to all blocks if no indexer key

        stacked_keys = torch.stack([b.indexer_key for b in blocks], dim=0).unsqueeze(0)  # (1, num_blocks, key_dim)
        key_mask = torch.ones(1, num_blocks, dtype=torch.bool, device=query_hidden.device)

        top_k_indices = self.indexer(query_hidden, stacked_keys, key_mask)
        top_k_idx = top_k_indices[0].tolist()

        return [blocks[idx] for idx in top_k_idx if idx >= 0]

    def get_dense(self, layer_idx: int) -> list[CompressedBlock]:
        """Tier 3: Return all compressed blocks for dense attention."""
        return self.hca_cache[layer_idx]

    def get_tail_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the uncompressed tail KV pairs for a layer.

        Returns:
            (keys, values) tensors of shape (tail_len, kv_heads, head_dim).
        """
        tail = self.tails[layer_idx]
        if not tail:
            return torch.tensor([]), torch.tensor([])
        # Handle 2-element (key, value) or 3-element (key, value, hidden_state) tuples
        keys = torch.stack([item[0] for item in tail], dim=0)
        values = torch.stack([item[1] for item in tail], dim=0)
        return keys, values

    @property
    def cache_stats(self) -> dict:
        """Return cache statistics for monitoring."""
        stats = {
            "swa_buffer_sizes": [buf.size for buf in self.swa_buffers],
            "tail_sizes": [len(t) for t in self.tails],
            "csa_block_counts": [len(c) for c in self.csa_cache],
            "hca_block_counts": [len(c) for c in self.hca_cache],
            "total_tokens": self.token_counts,
        }
        return stats
