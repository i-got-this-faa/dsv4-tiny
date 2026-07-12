"""DSV4-Tiny Phase 1: Warm-start compression weights.

Trains compression modules (encoder/decoder, indexer) via LM loss.

Usage:
    uv run python scripts/warm_start.py --from-config rtx4050 --output outputs/warm_start/
    uv run python scripts/warm_start.py --from-config rtx4050 --dataset HuggingFaceTB/cosmopedia-100k
    uv run python scripts/warm_start.py --seq_len 128 --max_steps 500
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from accelerate import cpu_offload
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dsv4_tiny.config import DSV4TinyConfig
from dsv4_tiny.model import DSV4TinyForCausalLM
from dsv4_tiny.utils import get_linear_schedule_with_cosine_decay


class RandomTextDataset(Dataset):
    """Synthetic random token dataset — fallback."""

    def __init__(self, vocab_size=248044, seq_len=64, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, min(50000, self.vocab_size), (self.seq_len,))
        return {"input_ids": input_ids}


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
    return RandomTextDataset(
        vocab_size=248044,
        seq_len=args.seq_len,
        num_samples=args.num_samples,
    )


def warm_start_loss(
    model: DSV4TinyForCausalLM,
    input_ids: torch.Tensor,
    cpu_offload_lm_head: bool = False,
    use_grad_checkpoint: bool = False,
) -> torch.Tensor:
    """Compute warm-start loss: autoencoder-style compression training.

    Forward through the model with cache enabled; LM loss trains compression modules.
    """
    outputs = model(
        input_ids,
        use_cache=True,
        labels=input_ids,
        cpu_offload_lm_head=cpu_offload_lm_head,
        use_grad_checkpoint=use_grad_checkpoint,
    )
    return outputs["loss"]


def main():
    parser = argparse.ArgumentParser(
        description="DSV4-Tiny Phase 1: Warm-start compression weights"
    )
    parser.add_argument(
        "--from-config",
        type=str,
        default=None,
        help="Load training preset from configs/<name>.toml (e.g. rtx4050, t4-colab). "
        "Overrides individual CLI flags.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HuggingFace dataset path (e.g. HuggingFaceTB/cosmopedia-100k). "
        "When set, replaces random data with real text.",
    )
    parser.add_argument("--output", type=str, default="outputs/warm_start")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=64,
        help="Sequence length per sample. 64-128 recommended for 6GB GPU.",
    )
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--cpu_offload_lm_head",
        action="store_true",
        default=True,
        help="Offload LM head to CPU during forward to save GPU memory",
    )
    parser.add_argument(
        "--grad_checkpoint",
        action="store_true",
        default=False,
        help="Gradient checkpointing per decoder layer.",
    )
    args = parser.parse_args()

    # Load preset if --from-config is given (overrides defaults above)
    if args.from_config:
        from dsv4_tiny.utils import load_training_config

        preset = load_training_config(args.from_config)
        for key, val in preset.items():
            if hasattr(args, key):
                setattr(args, key, val)
                print(f"  [config] {key} = {val}")
            elif key in ("use_grad_checkpoint",):
                setattr(args, "grad_checkpoint", val)
            # ignore unknown keys silently

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Config
    cfg = DSV4TinyConfig()

    # Model — initialize from scratch (no pretrained weights for warm-start)
    model = DSV4TinyForCausalLM(cfg)
    model = model.to(device=device, dtype=torch.bfloat16)

    # Freeze base model weights (embeddings, MLPs, norms, LM head)
    for name, param in model.named_parameters():
        if any(
            skip in name
            for skip in [
                "embed_tokens",
                "mlp",
                "lm_head",
                "norm",
                "input_layernorm",
                "post_attention_layernorm",
            ]
        ):
            param.requires_grad = False
        elif "self_attn" in name:
            # Only train compression-specific weights in attention
            if any(k in name for k in ["W_DQ", "W_UQ", "o_proj", "o_gate"]):
                param.requires_grad = True
            elif "sink_token" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Ensure compression modules (cache) are trainable
    for name, param in model.cache.named_parameters():
        param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.1f}%)")

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
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

    progress_bar = tqdm(total=args.max_steps, desc="Warm-start")
    running_loss = 0.0

    while global_step < args.max_steps:
        for batch in dataloader:
            if global_step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device)

            loss = warm_start_loss(
                model,
                input_ids,
                cpu_offload_lm_head=args.cpu_offload_lm_head,
                use_grad_checkpoint=args.grad_checkpoint,
            )
            loss = loss / args.grad_accum
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
                progress_bar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )
                running_loss = 0.0

            if global_step % args.save_interval == 0:
                # Save compression weights
                checkpoint = {
                    "compression_weights": {
                        k: v.clone().cpu()
                        for k, v in model.cache.state_dict().items()
                    },
                    "attention_weights": {
                        k: v.clone().cpu()
                        for k, v in model.state_dict().items()
                        if any(
                            s in k
                            for s in [
                                "W_DQ",
                                "W_UQ",
                                "o_proj",
                                "o_gate",
                                "sink_token",
                            ]
                        )
                    },
                    "config": cfg.to_dict(),
                    "step": global_step,
                }
                ckpt_path = output_dir / f"checkpoint-{global_step}.pt"
                torch.save(checkpoint, ckpt_path)
                print(f"\nSaved checkpoint to {ckpt_path}")

    progress_bar.close()

    # Save final weights
    final_ckpt = {
        "compression_weights": {
            k: v.clone().cpu() for k, v in model.cache.state_dict().items()
        },
        "attention_weights": {
            k: v.clone().cpu()
            for k, v in model.state_dict().items()
            if any(
                s in k
                for s in ["W_DQ", "W_UQ", "o_proj", "o_gate", "sink_token"]
            )
        },
        "config": cfg.to_dict(),
        "step": global_step,
    }
    final_path = output_dir / "compression_weights.pt"
    torch.save(final_ckpt, final_path)
    print(f"Final weights saved to {final_path}")

    print("Warm-start complete!")


if __name__ == "__main__":
    main()
