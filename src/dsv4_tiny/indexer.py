"""Lightning Indexer for sparse block retrieval.

Implements Equations 13-17 from the DSv4 paper:
    cQ = h @ W_DQ        (shared decomposed query)
    qI = cQ @ W_IUQ      (indexer query up-projection)
    wI = h @ W_w         (head-wise scoring weights)
    I_s = sum_h wI_h * ReLU(qI_h @ KI_Comp_s)    (score per block)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DSV4TinyConfig


class LightningIndexer(nn.Module):
    """Lightweight indexer that scores compressed KV blocks for sparse retrieval.

    For a given query hidden state, computes scores over all compressed blocks
    in the cache and returns indices of the top-k blocks.
    """

    def __init__(self, config: DSV4TinyConfig):
        super().__init__()
        d = config.hidden_size             # 1024
        n_Ih = config.indexer_heads        # 8
        c_I = config.indexer_dim           # 128
        self.top_k = config.indexer_top_k  # 128

        # W_DQ: (d, c_I * n_Ih) — decomposed query projection
        self.W_DQ = nn.Linear(d, c_I * n_Ih, bias=False)

        # W_IUQ: (c_I, c_I) — indexer query up-projection per head
        self.W_IUQ = nn.Linear(c_I, c_I, bias=False)

        # W_w: (d, n_Ih) — head-wise scoring weights
        self.W_w = nn.Linear(d, n_Ih, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.W_DQ.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_IUQ.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_w.weight, mean=0.0, std=0.02)

    def forward(
        self,
        query_hidden: torch.Tensor,
        compressed_keys: torch.Tensor,
        compressed_key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score compressed blocks and return top-k indices.

        Args:
            query_hidden: (batch_size, d) — current token's hidden state for query.
            compressed_keys: (batch_size, num_blocks, key_dim) — KI_Comp for each block.
                key_dim = c_I * n_Ih (indexer dim * indexer heads)
            compressed_key_mask: (batch_size, num_blocks) — boolean mask for valid blocks.

        Returns:
            top_k_indices: (batch_size, top_k) — indices of the top-k scored blocks.
        """
        b, num_blocks, key_dim = compressed_keys.shape
        n_Ih = self.W_w.out_features
        c_I = self.W_IUQ.in_features

        # 1. Decomposed query: cQ = h @ W_DQ
        cQ = self.W_DQ(query_hidden)  # (b, c_I * n_Ih)

        # 2. Reshape to per-head: (b, n_Ih, c_I)
        cQ = cQ.view(b, n_Ih, c_I)

        # 3. Indexer query up-projection per head: qI = cQ @ W_IUQ
        qI = self.W_IUQ(cQ)  # (b, n_Ih, c_I)

        # 4. Head weights: wI = h @ W_w
        wI = self.W_w(query_hidden)  # (b, n_Ih)

        # 5. Reshape compressed_keys to per-head: (b, n_Ih, num_blocks, c_I)
        KI = compressed_keys.view(b, num_blocks, n_Ih, c_I).transpose(1, 2)  # (b, n_Ih, num_blocks, c_I)

        # 6. Compute per-head scores: qI @ KI^T  → (b, n_Ih, num_blocks)
        head_scores = torch.matmul(qI.unsqueeze(2), KI.transpose(-1, -2)).squeeze(2)  # (b, n_Ih, num_blocks)

        # 7. Apply ReLU
        head_scores = F.relu(head_scores)  # (b, n_Ih, num_blocks)

        # 8. Weighted sum across heads: I_s = sum_h wI_h * ReLU(...)
        block_scores = (wI.unsqueeze(-1) * head_scores).sum(dim=1)  # (b, num_blocks)

        # 9. Mask invalid blocks (use logical_not for boolean safety)
        block_scores = block_scores.masked_fill(compressed_key_mask.logical_not(), float("-inf"))

        # 10. Return top-k indices
        actual_k = min(self.top_k, num_blocks)
        top_k_indices = torch.topk(block_scores, k=actual_k, dim=-1).indices  # (b, actual_k)

        # Pad to top_k if fewer blocks available
        if actual_k < self.top_k:
            padding = torch.full((b, self.top_k - actual_k), -1, device=top_k_indices.device, dtype=torch.long)
            top_k_indices = torch.cat([top_k_indices, padding], dim=-1)

        return top_k_indices
