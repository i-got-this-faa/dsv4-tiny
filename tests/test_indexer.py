import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.indexer import LightningIndexer


def test_indexer_output_shape():
    cfg = DSV4TinyConfig()
    indexer = LightningIndexer(cfg)

    b = 2
    d = cfg.hidden_size
    num_blocks = 64
    key_dim = cfg.indexer_dim * cfg.indexer_heads  # 128 * 8 = 1024

    query = torch.randn(b, d)
    compressed_keys = torch.randn(b, num_blocks, key_dim)
    key_mask = torch.ones(b, num_blocks, dtype=torch.bool)

    top_k = indexer(query, compressed_keys, key_mask)
    assert top_k.shape == (b, cfg.indexer_top_k), \
        f"Expected ({b}, {cfg.indexer_top_k}), got {top_k.shape}"


def test_indexer_top_k_less_than_blocks():
    cfg = DSV4TinyConfig()
    indexer = LightningIndexer(cfg)

    b = 1
    d = cfg.hidden_size
    num_blocks = 10  # Less than top_k=128
    key_dim = cfg.indexer_dim * cfg.indexer_heads

    query = torch.randn(b, d)
    compressed_keys = torch.randn(b, num_blocks, key_dim)
    key_mask = torch.ones(b, num_blocks, dtype=torch.bool)

    top_k = indexer(query, compressed_keys, key_mask)
    # Should pad to top_k with -1
    assert top_k.shape == (b, cfg.indexer_top_k)
    # First num_blocks should be valid, rest should be -1
    assert (top_k[0, :num_blocks] >= 0).all()
    assert (top_k[0, num_blocks:] == -1).all()


def test_indexer_masked_blocks():
    cfg = DSV4TinyConfig()
    indexer = LightningIndexer(cfg)

    b = 1
    d = cfg.hidden_size
    num_blocks = 16
    key_dim = cfg.indexer_dim * cfg.indexer_heads

    query = torch.randn(b, d)
    compressed_keys = torch.randn(b, num_blocks, key_dim)
    # Mask out the first 8 blocks
    key_mask = torch.tensor([[False] * 8 + [True] * 8])

    top_k = indexer(query, compressed_keys, key_mask)
    # There are 8 valid blocks (indices 8-15) and they should be top-ranked
    # before any -inf (masked) entries
    top_8 = top_k[0, :8]
    assert (top_8 >= 8).all(), f"Top 8 entries should be valid blocks >= 8, got {top_8}"

def test_indexer_gradient_flow():
    cfg = DSV4TinyConfig()
    indexer = LightningIndexer(cfg)

    b = 1
    d = cfg.hidden_size
    num_blocks = 8
    key_dim = cfg.indexer_dim * cfg.indexer_heads

    query = torch.randn(b, d, requires_grad=True)
    compressed_keys = torch.randn(b, num_blocks, key_dim)
    key_mask = torch.ones(b, num_blocks, dtype=torch.bool)

    # Forward through the full indexer forward, then use block_scores directly
    # (the topk operation is non-differentiable, but the scores before topk are)
    b, num_blocks, key_dim = compressed_keys.shape
    n_Ih = indexer.W_w.out_features
    c_I = indexer.W_IUQ.in_features

    cQ = indexer.W_DQ(query)
    cQ = cQ.view(b, n_Ih, c_I)
    qI = indexer.W_IUQ(cQ)
    wI = indexer.W_w(query)
    KI = compressed_keys.view(b, num_blocks, n_Ih, c_I).transpose(1, 2)
    head_scores = torch.matmul(qI.unsqueeze(2), KI.transpose(-1, -2)).squeeze(2)
    head_scores = F.relu(head_scores)
    block_scores = (wI.unsqueeze(-1) * head_scores).sum(dim=1)

    loss = block_scores.sum()
    loss.backward()

    assert indexer.W_DQ.weight.grad is not None, "W_DQ should get gradients"
    assert indexer.W_IUQ.weight.grad is not None, "W_IUQ should get gradients"
    assert indexer.W_w.weight.grad is not None, "W_w should get gradients"
