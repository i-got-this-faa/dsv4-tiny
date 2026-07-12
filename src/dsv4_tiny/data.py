"""Dataset utilities for DSV4-Tiny training."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, IterableDataset


class RandomTextDataset(Dataset):
    """Synthetic random token dataset — fallback when no real data is configured."""

    def __init__(self, vocab_size: int = 248044, seq_len: int = 64, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class TokenizedTextDataset(IterableDataset):
    """Streams text from a HuggingFace dataset, tokenizes, and chunks to seq_len.

    Uses HuggingFace ``datasets`` with ``streaming=True`` so you don't need
    to download the full dataset upfront.

    Args:
        hf_path: HuggingFace dataset path (e.g. ``"HuggingFaceTB/cosmopedia-100k"``).
        split: Dataset split (default ``"train"``).
        text_key: Column name containing the text (default ``"text"``).
        tokenizer: A ``transformers.PreTrainedTokenizer`` instance.
        seq_len: Chunk size in tokens.
        max_samples: Maximum number of chunks to yield (for quick testing).
    """

    def __init__(
        self,
        hf_path: str,
        split: str = "train",
        text_key: str = "text",
        tokenizer=None,
        seq_len: int = 64,
        max_samples: int | None = None,
    ):
        super().__init__()
        from datasets import load_dataset

        self.ds = load_dataset(hf_path, split=split, streaming=True)
        self.text_key = text_key
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_samples = max_samples

    def __iter__(self):
        from transformers import AutoTokenizer

        tokenizer = self.tokenizer or AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
        buf = []
        count = 0

        for example in self.ds:
            text = example.get(self.text_key, "")
            if not text:
                continue

            ids = tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]

            buf.extend(ids)

            # Emit full chunks from the buffer
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                ids_t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": ids_t, "labels": ids_t.clone()}
                count += 1
                if self.max_samples and count >= self.max_samples:
                    return

    def __len__(self) -> int:
        # IterableDataset needs this for some schedulers; estimate
        return self.max_samples or 999999
