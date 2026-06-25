"""DSV4-Tiny benchmark: measure KV cache size vs uncompressed baseline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.model import DSV4TinyForCausalLM


def estimate_uncompressed_cache_size(seq_len: int, config: DSV4TinyConfig) -> int:
    """Estimate uncompressed KV cache size in bytes.

    Uncompressed: layers * 2 (K+V) * kv_heads * head_dim * seq_len * bytes_per_element
    """
    bytes_per_element = 2  # FP16
    size = (
        config.num_hidden_layers
        * 2  # K + V
        * config.num_key_value_heads
        * config.head_dim
        * seq_len
        * bytes_per_element
    )
    return size


def estimate_compressed_cache_size(seq_len: int, config: DSV4TinyConfig) -> int:
    """Estimate DSV4 compressed KV cache size in bytes.

    - SWA layers (2): uncompressed, n_win=128 tokens each
    - CSA layers (11): compressed blocks of m=4
    - HCA layers (11): compressed blocks of m'=128
    """
    bytes_per_element = 2  # FP16

    # SWA: uncompressed sliding window (128 tokens)
    swa_size = (
        len(config.swa_layers)
        * 2  # K+V
        * config.num_key_value_heads
        * config.head_dim
        * config.swa_window_size
        * bytes_per_element
    )

    # CSA: compressed blocks
    csa_compressed_dim = config.csa_compressed_dim * config.csa_groups + config.csa_intermediate
    num_csa_blocks = seq_len // config.csa_block_size
    csa_size = (
        len(config.csa_layers)
        * 2  # K+V
        * csa_compressed_dim
        * bytes_per_element
        * num_csa_blocks
    )

    # HCA: compressed blocks
    hca_compressed_dim = config.hca_compressed_dim
    num_hca_blocks = seq_len // config.hca_block_size
    hca_size = (
        len(config.hca_layers)
        * 2  # K+V
        * hca_compressed_dim
        * bytes_per_element
        * num_hca_blocks
    )

    # Uncompressed tail (up to block_alignment=128 per layer)
    tail_size = (
        config.num_hidden_layers
        * 2  # K+V
        * config.num_key_value_heads
        * config.head_dim
        * config.block_alignment
        * bytes_per_element
    )

    return swa_size + csa_size + hca_size + tail_size


@torch.no_grad()
def benchmark(context_lengths: list[int], device: torch.device) -> dict:
    """Run benchmark for different context lengths."""
    print(f"Benchmarking on {device}")
    print(f"{'Context':>10} {'Uncompressed':>14} {'Compressed':>14} {'Ratio':>8} {'Time (ms)':>10}")

    cfg = DSV4TinyConfig()
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    results = {}
    for seq_len in context_lengths:
        # Measure wall-clock time for a forward pass
        input_ids = torch.randint(0, 1000, (1, min(seq_len, 2048)), device=device)

        torch.cuda.synchronize() if device.type == "cuda" else None
        start = time.time()

        _ = model(input_ids, use_cache=True)

        torch.cuda.synchronize() if device.type == "cuda" else None
        elapsed_ms = (time.time() - start) * 1000

        # Estimate sizes
        uncompressed_size = estimate_uncompressed_cache_size(seq_len, cfg)
        compressed_size = estimate_compressed_cache_size(seq_len, cfg)
        ratio = compressed_size / uncompressed_size if uncompressed_size > 0 else 0

        print(
            f"{seq_len:>10} "
            f"{uncompressed_size / 1024 / 1024:>8.1f} MB "
            f"{compressed_size / 1024 / 1024:>8.1f} MB "
            f"{ratio:>7.2%} "
            f"{elapsed_ms:>10.1f}"
        )

        results[seq_len] = {
            "uncompressed_mb": uncompressed_size / 1024 / 1024,
            "compressed_mb": compressed_size / 1024 / 1024,
            "ratio": ratio,
            "time_ms": elapsed_ms,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="DSV4-Tiny benchmark")
    parser.add_argument(
        "--context-lengths",
        type=str,
        default="2048,8192,32768",
        help="Comma-separated context lengths",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    benchmark(context_lengths, device)


if __name__ == "__main__":
    main()
