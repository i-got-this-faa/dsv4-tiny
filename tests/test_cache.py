"""Tests for DSV4Cache."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.cache import DSV4Cache


def test_cache_push_and_tail():
    cfg = DSV4TinyConfig()
    cache = DSV4Cache(cfg)

    # Push a single KV pair
    kv = torch.randn(cfg.num_key_value_heads, cfg.head_dim)
    for layer_idx in range(3):
        cache.push(layer_idx, kv, kv)

    # Check tail
    for layer_idx in range(3):
        tail_k, tail_v = cache.get_tail_kv(layer_idx)
        assert tail_k.shape[0] == 1, f"Tail should have 1 element, got {tail_k.shape[0]}"


def test_cache_compression_trigger():
    cfg = DSV4TinyConfig()
    cache = DSV4Cache(cfg)

    block_size = cfg.block_alignment  # 128
    kv = torch.randn(cfg.num_key_value_heads, cfg.head_dim)

    # Push exactly block_size tokens to trigger compression
    for i in range(block_size):
        cache.push(2, kv, kv)  # Layer 2 is CSA

    # After pushing block_size tokens, tail should be empty
    tail_k, _ = cache.get_tail_kv(2)
    assert tail_k.numel() == 0 or tail_k.shape[0] < block_size, \
        f"Tail should have < {block_size} elements after compression"


def test_cache_state_window():
    cfg = DSV4TinyConfig()
    cache = DSV4Cache(cfg)

    # Push tokens to a SWA layer
    kv = torch.randn(cfg.num_key_value_heads, cfg.head_dim)
    for i in range(10):
        cache.push(0, kv, kv)

    win_k, win_v = cache.get_state_window(0)
    assert win_k.shape[0] == 10, f"Window should have 10 elements, got {win_k.shape[0]}"


def test_cache_window_eviction():
    cfg = DSV4TinyConfig()
    cache = DSV4Cache(cfg)

    window = cfg.swa_window_size  # 128
    kv = torch.randn(cfg.num_key_value_heads, cfg.head_dim)

    # Push more than window size
    for i in range(window + 10):
        cache.push(0, kv, kv)

    win_k, _ = cache.get_state_window(0)
    assert win_k.shape[0] <= window, \
        f"Window should be at most {window}, got {win_k.shape[0]}"


def test_cache_reset():
    cfg = DSV4TinyConfig()
    cache = DSV4Cache(cfg)

    kv = torch.randn(cfg.num_key_value_heads, cfg.head_dim)
    for i in range(5):
        cache.push(0, kv, kv)

    cache.reset()

    tail_k, _ = cache.get_tail_kv(0)
    assert tail_k.numel() == 0, "Tail should be empty after reset"
    win_k, _ = cache.get_state_window(0)
    assert win_k.numel() == 0, "Window should be empty after reset"


def test_cache_csa_block_count():
    cfg = DSV4TinyConfig()
    cache = DSV4Cache(cfg)

    block_size = cfg.block_alignment
    kv = torch.randn(cfg.num_key_value_heads, cfg.head_dim)

    # Push 3 blocks worth to a CSA layer
    for i in range(block_size * 3):
        cache.push(2, kv, kv)

    blocks = cache.csa_cache[2]
    assert len(blocks) == 3, f"Expected 3 compressed blocks, got {len(blocks)}"


def test_cache_hca_block_count():
    cfg = DSV4TinyConfig()
    cache = DSV4Cache(cfg)

    block_size = cfg.block_alignment
    kv = torch.randn(cfg.num_key_value_heads, cfg.head_dim)

    # Push 2 blocks worth to an HCA layer
    for i in range(block_size * 2):
        cache.push(3, kv, kv)

    blocks = cache.hca_cache[3]
    assert len(blocks) == 2, f"Expected 2 compressed blocks, got {len(blocks)}"
