#!/usr/bin/env python3
"""DSV4-Tiny Phase 2: LoRA fine-tune.

Fine-tunes the full model with compression weights frozen and LoRA adapters
on attention projections.

Usage:
    uv run python scripts/finetune_lora.py --checkpoint outputs/warm_start/compression_weights.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim import AdamW
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.model import DSV4TinyForCausalLM
from dsv4_tiny.utils import get_linear_schedule_with_cosine_decay


class LoRALayer(nn.Module):
    """Simple LoRA adapter layer (r=16, alpha=16)."""

    def __init__(self, in_dim: int, out_dim: int, r: int = 16, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(in_dim, r) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(r, out_dim))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling


def add_lora_to_model(
    model: DSV4TinyForCausalLM,
    r: int = 16,
    alpha: int = 16,
    dropout: float = 0.0,
) -> None:
    """Add LoRA adapters to Q/K/V/O projections in attention layers."""
    lora_modules = {}
    for name, module in model.named_modules():
        if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            if isinstance(module, nn.Linear) and module.weight.requires_grad is False:
                # Store LoRA adapter on the module
                lora = LoRALayer(
                    module.in_features, module.out_features,
                    r=r, alpha=alpha, dropout=dropout,
                )
                module.lora = lora
                lora_modules[name] = lora

    print(f"Added LoRA adapters to {len(lora_modules)} linear layers")
    return lora_modules


def make_dataset(args):
    """Build training dataset — real HuggingFace data or random fallback."""
    if args.dataset:
        from dsv4_tiny.data import TokenizedTextDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        return TokenizedTextDataset(
            hf_path=args.dataset,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            max_samples=args.num_samples,
        )
    # Fallback: random token data
    class _Random(Dataset):
        def __init__(self, seq_len, num_samples):
            self.seq_len = seq_len
            self.num_samples = num_samples
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            input_ids = torch.randint(0, 50000, (self.seq_len,))
            return {"input_ids": input_ids, "labels": input_ids.clone()}
    return _Random(seq_len=args.seq_len, num_samples=args.num_samples)


def main():
    parser = argparse.ArgumentParser(description="DSV4-Tiny Phase 2: LoRA fine-tune")
    parser.add_argument("--from-config", type=str, default=None,
                        help="Load training preset from configs/<name>.toml (e.g. rtx4050, t4-colab). "
                             "Overrides individual CLI flags.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="HuggingFace dataset path (e.g. HuggingFaceTB/cosmopedia-100k). "
                             "When set, replaces random data with real text.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to warm-start compression_weights.pt")
    parser.add_argument("--output", type=str, default="outputs/lora")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=64,
                        help="Sequence length per sample. 64-128 recommended for 6GB GPU.")
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cpu_offload_lm_head", action="store_true", default=True,
                        help="Offload LM head to CPU during forward to save GPU memory")
    args = parser.parse_args()

    # Load preset if --from-config is given (overrides defaults above)
    if args.from_config:
        from dsv4_tiny.utils import load_training_config
        preset = load_training_config(args.from_config)
        for key, val in preset.items():
            if hasattr(args, key):
                setattr(args, key, val)
                print(f"  [config] {key} = {val}")
            # ignore unknown keys silently

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Config
    cfg = DSV4TinyConfig()

    # Model
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(device=device, dtype=torch.bfloat16)

    # Load warm-start compression weights
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading compression weights from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

        if "compression_weights" in checkpoint:
            model.cache.load_state_dict(checkpoint["compression_weights"])

        if "attention_weights" in checkpoint:
            # Load attention-specific weights
            attn_state = checkpoint["attention_weights"]
            model_state = model.state_dict()
            for k, v in attn_state.items():
                if k in model_state:
                    model_state[k].copy_(v.to(dtype=model_state[k].dtype))

    # Freeze everything except LoRA adapters
    for param in model.parameters():
        param.requires_grad = False

    # Add LoRA adapters
    lora_modules = add_lora_to_model(
        model, r=args.lora_r, alpha=args.lora_alpha, dropout=0.0,
    )

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable LoRA params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Optimizer (LoRA params only)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )

    # LR scheduler
    scheduler = get_linear_schedule_with_cosine_decay(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
    )
    # Dataset
    dataset = make_dataset(args)
    is_iterable = isinstance(dataset, IterableDataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not is_iterable,
        drop_last=True,
    )

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    global_step = 0
    optimizer.zero_grad()

    progress_bar = tqdm(total=args.max_steps, desc="LoRA fine-tune")
    running_loss = 0.0

    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels, use_cache=True,
                            cpu_offload_lm_head=args.cpu_offload_lm_head)
            loss = outputs["loss"] / args.grad_accum
            loss.backward()

            running_loss += loss.item() * args.grad_accum

            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)

            if global_step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })
                running_loss = 0.0

            if global_step % args.save_interval == 0:
                # Save LoRA adapters
                lora_state = {
                    k: v.clone().cpu()
                    for k, v in model.state_dict().items()
                    if "lora" in k
                }
                ckpt_path = output_dir / f"lora-checkpoint-{global_step}.pt"
                torch.save(lora_state, ckpt_path)
                print(f"\nSaved LoRA checkpoint to {ckpt_path}")

    progress_bar.close()

    # Save final LoRA adapters
    lora_state = {
        k: v.clone().cpu()
        for k, v in model.state_dict().items()
        if "lora" in k
    }
    final_path = output_dir / "adapter_model.pt"
    torch.save(lora_state, final_path)
    print(f"Final LoRA adapters saved to {final_path}")

    print("LoRA fine-tune complete!")


if __name__ == "__main__":
    main()
